
import os
import sys
import copy
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision.transforms import ToTensor

# Dossier du projet
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))

import test_base as tb

# Chemins
CHECKPOINT = BASE_DIR / "checkpoints" / "base" / "checkpoint_base_1.pth.tar"
IMAGE_PATH = BASE_DIR / "videos_5" / "input_png" / "frame_001.png"
OUTPUT_IMAGE = BASE_DIR / "iframe_test_reconstruction.png"

print("========== TEST I-FRAME BITSTREAM ==========")
print("Device :", tb.device)

# 1. Construire les trois codeurs comme dans le programme original
def load_coder(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    architecture = tb.__CODER_TYPES__[cfg["model_architecture"]]
    return architecture(**cfg["model_params"])

mo_coder = load_coder(BASE_DIR / "config" / "DVC_motion.yml")
cond_mo_coder = load_coder(BASE_DIR / "config" / "CANF_motion_predprior.yml")
res_coder = load_coder(BASE_DIR / "config" / "CANF_inter_coder.yml")

net = tb.Pframe(mo_coder, cond_mo_coder, res_coder).to(tb.device)

# 2. Ajouter YOLO comme dans le chargement original
yolo = tb.DetectMultiBackend(
    weights=BASE_DIR / "yolov5s.pt",
    device=tb.device,
    dnn=False,
    data=BASE_DIR / "data" / "hevc_class_b_coded.yaml",
    fp16=False
)

net.feature_extractor = copy.deepcopy(yolo)

# 3. Charger le checkpoint original
checkpoint = torch.load(CHECKPOINT, map_location=tb.device)
state_dict = checkpoint["state_dict"]

state_dict = {
    key.replace("module.", "", 1) if key.startswith("module.") else key: value
    for key, value in state_dict.items()
}

net.load_state_dict(state_dict, strict=True)
net.eval()

print("Checkpoint chargé avec strict=True.")

# 4. Lire une seule image
image = Image.open(IMAGE_PATH).convert("RGB")
original = ToTensor()(image).unsqueeze(0).to(tb.device)

print("Image originale :", tuple(original.shape))

# 5. Aligner l'image
align = tb.Alignment().to(tb.device)
aligned = align.align(original)

print("Image alignée :", tuple(aligned.shape))

# 6. Encodage binaire réel
with torch.no_grad():
    strings, shapes = net.if_model.compress(
        aligned,
        return_hat=False
    )

print("\nEncodage terminé.")

# Calculer la taille des flux binaires
def count_bytes(obj):
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, (list, tuple)):
        return sum(count_bytes(item) for item in obj)
    raise TypeError("Type de flux non reconnu : " + str(type(obj)))

main_bytes = count_bytes(strings[0])
side_bytes = count_bytes(strings[1])
total_bytes = main_bytes + side_bytes

print("Flux principal :", main_bytes, "octets")
print("Flux secondaire :", side_bytes, "octets")
print("Taille totale :", total_bytes, "octets")
print("Taille totale :", total_bytes / 1024, "Ko")

# 7. Décodage binaire réel
with torch.no_grad():
    reconstructed = net.if_model.decompress(strings, shapes)

reconstructed = align.resume(reconstructed).clamp(0, 1)

print("\nDécodage terminé.")
print("Image reconstruite :", tuple(reconstructed.shape))

# 8. Vérifier la reconstruction
mse = torch.mean((original - reconstructed) ** 2).item()

if mse > 0:
    psnr = 10 * torch.log10(torch.tensor(1.0 / mse)).item()
else:
    psnr = float("inf")

print("MSE :", mse)
print("PSNR :", psnr, "dB")

# 9. Sauvegarder l'image reconstruite
tb.torch2img(reconstructed.cpu()).save(OUTPUT_IMAGE)

print("\nImage reconstruite sauvegardée :", OUTPUT_IMAGE)
print("========== TEST TERMINE ==========")
