import os, sys, copy, time, csv
from pathlib import Path
import torch, yaml
from PIL import Image
from torchvision.transforms import ToTensor
from codecarbon import EmissionsTracker

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))
import test_base as tb

# ---- Réglages ----
INDICES = [0, 33, 65, 97, 129]         # mets [0, 33, 65, 97, 129] pour les 5 I-frames
REPEATS = 3            # mets 3 pour moyenner (chaque repetition = environ 3 min par image)
CHECKPOINT = BASE_DIR / "checkpoints" / "base" / "checkpoint_base_1.pth.tar"
RESULTS = BASE_DIR / "energy_iframe_results.csv"   # nouveau fichier, n'ecrase rien

def load_coder(p):
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return tb.__CODER_TYPES__[cfg["model_architecture"]](**cfg["model_params"])

# ---- Chargement du modele (NON mesure) ----
net = tb.Pframe(load_coder(BASE_DIR / "config" / "DVC_motion.yml"),
                load_coder(BASE_DIR / "config" / "CANF_motion_predprior.yml"),
                load_coder(BASE_DIR / "config" / "CANF_inter_coder.yml")).to(tb.device)
yolo = tb.DetectMultiBackend(weights=BASE_DIR / "yolov5s.pt", device=tb.device, dnn=False,
                             data=BASE_DIR / "data" / "hevc_class_b_coded.yaml", fp16=False)
net.feature_extractor = copy.deepcopy(yolo)
ck = torch.load(CHECKPOINT, map_location=tb.device)
sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
      for k, v in ck["state_dict"].items()}
net.load_state_dict(sd, strict=True)
net.eval()
align = tb.Alignment().to(tb.device)

def count_bytes(o):
    if isinstance(o, (bytes, bytearray)): return len(o)
    return sum(count_bytes(i) for i in o)

def measure(label, fn):
    tr = EmissionsTracker(project_name=label, measure_power_secs=5,
                          save_to_file=False, log_level="error")
    tr.start()
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    tr.stop()
    e_mwh = tr.final_emissions_data.energy_consumed * 1e6   # kWh -> mWh
    return out, dt, e_mwh

new = not RESULTS.exists()
f = open(RESULTS, "a", newline="", encoding="utf-8")
w = csv.writer(f)
if new:
    w.writerow(["index", "frame", "run", "enc_time_s", "enc_mWh", "dec_time_s", "dec_mWh",
                "bytes_main", "bytes_side", "n_clamped", "mse", "psnr_db"])

for idx in INDICES:
    name = f"frame_{idx + 1:03d}.png"
    img = Image.open(BASE_DIR / "videos_5" / "input_png" / name).convert("RGB")
    original = ToTensor()(img).unsqueeze(0).to(tb.device)
    aligned = align.align(original)

    for run in range(1, REPEATS + 1):
        def enc():
            with torch.no_grad():
                return net.if_model.compress(aligned, return_hat=False)
        (strings, shapes), t_enc, e_enc = measure(f"enc_{idx}_{run}", enc)

        def dec():
            with torch.no_grad():
                return net.if_model.decompress(strings, shapes)
        rec, t_dec, e_dec = measure(f"dec_{idx}_{run}", dec)

        rec = align.resume(rec).clamp(0, 1)
        mse = torch.mean((original - rec) ** 2).item()
        psnr = 10 * torch.log10(torch.tensor(1.0 / mse)).item() if mse > 0 else float("inf")
        n_clamp = getattr(getattr(net.if_model, "entropy_bottleneck", None), "last_n_clamped", -1)

        row = [idx, name, run, round(t_enc, 2), round(e_enc, 3), round(t_dec, 2), round(e_dec, 3),
               count_bytes(strings[0]), count_bytes(strings[1]), n_clamp, mse, round(psnr, 3)]
        w.writerow(row); f.flush()
        print(f"[{name} run {run}] ENC {t_enc:.1f}s {e_enc:.2f}mWh | DEC {t_dec:.1f}s {e_dec:.2f}mWh | "
              f"{row[7]}+{row[8]} o | clamp={n_clamp} | MSE={mse:.6e} PSNR={psnr:.3f}")
f.close()
print("Resultats ajoutes a", RESULTS)