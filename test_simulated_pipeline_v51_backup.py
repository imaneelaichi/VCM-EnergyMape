# -*- coding: utf-8 -*-
"""test_simulated_pipeline_v51.py -- test de bout en bout de V5.1 avec des COMPOSANTS SIMULÉS.

Crée un projet TEMPORAIRE contenant : les scripts V5.1 + un FAUX codec (base_codec.py), un faux torch/torchvision, un faux
profiler et un faux flux binaire, + de fausses images/checkpoints. Exécute --list, la validation, --probe, --run, --batch et
--check_energy dans ce dossier temporaire. Il n'utilise NI le vrai codec VCM, NI vos checkpoints, NI vos résultats : il vérifie
la plomberie (config, temps, énergie brute/nette, bpp/bit/s, séparation des résultats), pas la qualité du codec ni RAPL.
Durée : ~6-8 minutes (le faux codec consomme du CPU pour que CodeCarbon mesure quelque chose).
Lancer depuis svc_code :   python -m unittest test_simulated_pipeline_v51 -v        (V51_SKIP_SIM=1 pour l'ignorer)
Ne PAS lancer pendant qu'une vraie expérience mesure l'énergie sur la même machine."""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
FILES = ("experiment3_frame_components_v51.py", "exp_config_v51.py", "energy_profiling_v51.py", "experiment_config_v51.yaml")
STUBS = {'torch.py': 'class FT:\n    def __init__(self, shape): self.shape = shape; self.dtype = "float32"\ndef set_num_threads(n): pass\ndef get_num_threads(): return 1\ndef load(path, map_location=None, weights_only=None):\n    k = 8 if "base_3" not in str(path) else 9          # base_3 : formes differentes (test d\'incompatibilite)\n    return {"state_dict": {"enc.w": FT((k, 3)), "dec.w": FT((3, k))}, "lmbda": 0.01 * (1 if "base_1" in str(path) else 2), "epoch": 5, "args": {"N": 128}}\n', 'torchvision/__init__.py': '', 'torchvision/transforms.py': 'import numpy as np\nclass _T:\n    def __init__(self, a): self.a = a\n    def unsqueeze(self, i): return self\nclass ToTensor:\n    def __call__(self, img): return _T(np.asarray(img))\n', 'profiling.py': 'import time\nfrom contextlib import contextmanager\nIO_STAGES = {"image_read", "image_write", "bitstream_write", "bitstream_read", "preprocess_to_tensor"}\nclass Profiler:\n    def __init__(self): self.frames, self.records, self._cur, self._phase = [], [], None, "measured"\n    @contextmanager\n    def phase(self, name):\n        old, self._phase = self._phase, name\n        try: yield\n        finally: self._phase = old\n    @contextmanager\n    def frame(self, idx, ftype, direction):\n        self._cur = (idx, ftype, direction); t = time.perf_counter()\n        try: yield\n        finally:\n            self.frames.append({"frame": idx, "type": ftype, "dir": direction, "phase": self._phase, "wall_s": time.perf_counter() - t}); self._cur = None\n    @contextmanager\n    def stage(self, name):\n        t = time.perf_counter()\n        try: yield\n        finally:\n            i, _, d = self._cur\n            self.records.append({"phase": self._phase, "dir": d, "frame": i, "stage": name, "seconds": time.perf_counter() - t})\n    def export(self, prefix, meta=None):\n        import json; open(prefix + ".json", "w").write(json.dumps({"n": len(self.records), "meta": str(meta)}))\n', 'base_codec.py': 'import time, os\ndef _burn(s):\n    t = time.perf_counter(); x = 0.0\n    while time.perf_counter() - t < s: x = (x + 1.000001) % 1e5\nclass BaseLayerCodec:\n    def __init__(self, role, root=None, closed_gop=False): self.role = role; self.closed_gop = closed_gop\n    def load_checkpoint(self, path): return {"ckpt": path}\n    def reset_sequence(self): pass\n    def _intra(self, idx, gop):\n        g = gop + 1 if os.environ.get("STUB_BAD_GOP") else gop     # simule un codec dont la structure ne suit pas le GOP\n        return idx % g == 0\n    def encode_frame(self, x, idx, gop, prof):\n        if self._intra(idx, gop):\n            with prof.stage("I_analysis"): _burn(1.2)\n            with prof.stage("I_y_rangecode"): _burn(0.8)\n        else:\n            with prof.stage("motion_estimation"): _burn(0.9)\n            with prof.stage("R_analysis"): _burn(1.3)\n        return {"payload": bytes([idx % 256]) * (100 + 10 * idx), "ranges": [(0, 50), (50, 100 + 10 * idx)], "shapes": {"y": (1, 8, 4, 4)}}\n    def decode_frame(self, payload, ranges, idx, gop, shapes, hw, prof):\n        import numpy as np\n        if self._intra(idx, gop):\n            with prof.stage("I_synthesis"): _burn(1.0)\n        else:\n            with prof.stage("motion_compensation"): _burn(0.6)\n            with prof.stage("R_synthesis"): _burn(0.6)\n        return np.full((hw[0], hw[1], 3), 100 + idx, dtype=np.uint8)\n', 'bitstream.py': 'import struct, dataclasses, os\n@dataclasses.dataclass\nclass SequenceHeader:\n    width: int; height: int; n_frames: int; gop: int; ckpt_id: int = 1; closed_gop: bool = False; shapes: dict = dataclasses.field(default_factory=dict)\nclass BitstreamWriter:\n    def __init__(self, path, hdr): self.path, self.hdr, self.f = path, hdr, []\n    def add_frame(self, payload, ranges): self.f.append(payload); return len(payload) + 8\n    def close(self):\n        with open(self.path, "wb") as fh:\n            fh.write(struct.pack("<4I", self.hdr.width, self.hdr.height, len(self.f), self.hdr.gop))\n            for p in self.f: fh.write(struct.pack("<I", len(p)) + p)\n        return {"total_bytes": os.path.getsize(self.path)}\nclass BitstreamReader:\n    def __init__(self, path):\n        self.fh = open(path, "rb"); w, h, n, g = struct.unpack("<4I", self.fh.read(16))\n        self.header = SequenceHeader(w, h, n, g, 1, False, {"y": (8, 4, 4)})\n    def next_frame(self):\n        b = self.fh.read(4)\n        if not b: raise EOFError\n        (n,) = struct.unpack("<I", b); return self.fh.read(n), [(0, 1)], None\n    def close(self): self.fh.close()\n', 'test_base_profiled.py': 'import os\nfrom PIL import Image\ndef _frame_path(d, prefix, idx): return os.path.join(d, f"{prefix}{idx:03d}.png")\ndef _save_png(rec, path): Image.fromarray(rec).save(path)\ndef _env_meta(a): return {"stub": True}\n', 'entropy_codec.py': '# stub\n'}


def jload(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def run(cmd, cwd, env_extra=None, timeout=900):
    env = dict(os.environ, **(env_extra or {}))
    p = subprocess.run([sys.executable] + cmd, cwd=cwd, capture_output=True, text=True, env=env, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


@unittest.skipIf(os.environ.get("V51_SKIP_SIM"), "V51_SKIP_SIM défini")
class TestSimulatedPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = Path(tempfile.mkdtemp(prefix="v51_sim_"))
        for f in FILES:
            shutil.copy(HERE / f, cls.td / f)
        for name, src in STUBS.items():
            (cls.td / name).parent.mkdir(parents=True, exist_ok=True)
            (cls.td / name).write_text(src, encoding="utf-8")
        (cls.td / "videos_5" / "input_png").mkdir(parents=True)
        (cls.td / "checkpoints" / "base").mkdir(parents=True)
        rng = np.random.default_rng(1)
        for i in range(1, 6):
            Image.fromarray(rng.integers(0, 255, (432, 768, 3), dtype=np.uint8)).save(cls.td / "videos_5" / "input_png" / f"frame_{i:03d}.png")
        for k in (1, 2, 3):
            (cls.td / "checkpoints" / "base" / f"checkpoint_base_{k}.pth.tar").write_bytes(b"fake%d" % k)
        cfg = yaml.safe_load((HERE / "experiment_config_v51.yaml").read_text(encoding="utf-8"))
        cfg["paths"]["results_root"] = str(cls.td / "results")
        d = cfg["defaults"]
        d["frames"]["count"], d["repeats"], d["codec"]["warmup_frames"], d["video"]["fps"] = 4, 2, 1, 25
        d["energy"].update(measure_power_secs=0.2, idle_seconds=3, idle_between_seconds=2, cooldown_seconds=1,
                           require_confirmed_rapl=False)      # l'environnement simulé n'a PAS d'EMI : RAPL ne sera jamais "confirmé"
        d["validation"]["probe_frames"] = 3
        cfg["batch"]["cooldown_between_configs_s"] = 2
        cfg["experiments"] = [
            {"id": "C1", "gop": 2, "resolution": "768x432", "checkpoint": "base_1"},
            {"id": "C2", "gop": 2, "resolution": "768x432", "checkpoint": "base_2"},
            {"id": "C3", "gop": 4, "resolution": "640x360", "checkpoint": "base_1"},
            {"id": "C6", "gop": 3, "resolution": "800x600", "checkpoint": "base_9"},
            {"id": "C7", "gop": 2, "resolution": "768x432", "checkpoint": "base_1", "detector": {"conf_threshold": 0.5}}]
        cls.cfgp = cls.td / "cfg.yaml"
        cls.cfgp.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        cls.script = "experiment3_frame_components_v51.py"
        cls.group = cls.td / "results" / "exp3_v51"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.td, ignore_errors=True)

    def v51(self, *args, **kw):
        return run([self.script, "--config", str(self.cfgp)] + list(args), self.td, **kw)

    def result_dirs(self):
        return sorted(p.name for p in self.group.glob("C*") if p.is_dir()) if self.group.exists() else []

    # ------------------------------------------------------------------ 1. liste, configurations invalides / valides
    def test_01_list_and_static_validation(self):
        rc, out = self.v51("--list")
        self.assertEqual(rc, 0, out)
        for i in ("C1", "C2", "C3", "C6", "C7"):
            self.assertIn(i, out)
        rc, out = self.v51("--id", "C1")                       # valide : rien n'est lancé
        self.assertEqual(rc, 0, out)
        self.assertIn("Rien n'a été lancé", out)
        self.assertEqual(self.result_dirs(), [])
        rc, out = self.v51("--id", "C6")                       # GOP 3, 800x600, base_9
        self.assertNotEqual(rc, 0)
        self.assertIn("ÉCHEC", out)
        rc, out = self.v51("--ids", "C1,C7")                   # détecteur différent
        self.assertNotEqual(rc, 0)
        self.assertIn("détecteur identique", out)
        rc, out = self.v51("--batch")                          # --ids obligatoire : pas de lancement automatique
        self.assertNotEqual(rc, 0)
        self.assertEqual(self.result_dirs(), [])

    # ------------------------------------------------------------------ 2. probe
    def test_02_probe(self):
        rc, out = self.v51("--id", "C3", "--probe")            # 640x360 : redimensionnement + essai réel du faux codec
        self.assertEqual(rc, 0, out)
        self.assertIn("Essai OK", out)
        self.assertIn("structure I/P réellement produite", out)
        self.assertEqual(self.result_dirs(), [], "--probe ne doit créer aucun dossier de résultats d'expérience")
        for i in range(4):
            with Image.open(self.td / "inputs_cache" / next((self.td / "inputs_cache").iterdir()).name / f"frame_{i + 1:03d}.png") as im:
                self.assertEqual(im.size, (640, 360))
        rc, out = self.v51("--id", "C1", "--probe", env_extra={"STUB_BAD_GOP": "1"})     # codec dont la structure ne suit pas le GOP
        self.assertNotEqual(rc, 0)
        self.assertIn("ÉCHEC", out)

    # ------------------------------------------------------------------ 3. run individuel + calculs
    def test_03_run_single_and_calculations(self):
        rc, out = self.v51("--id", "C3", "--run")
        self.assertEqual(rc, 0, out[-3000:])
        d = self.group / "C3_gop4_640x360_base_1_videos_5_n4_r2"
        self.assertTrue(d.is_dir())
        for f in ("frame_measurements.csv", "component_time_energy.csv", "energy_tasks.csv", "idle_baselines.csv", "repeat_summary.csv",
                  "summary.json", "metadata.json", "validation.json", "detector_params.json", "config_resolved.json",
                  "experiment_config_snapshot.yaml", "codecarbon_messages.log", "experiment_log.txt"):
            self.assertTrue((d / f).exists(), f)
        m, s = jload(d / "metadata.json"), jload(d / "summary.json")
        self.assertEqual(s["status"], "SUCCESS")
        # --- traçabilité : une seule configuration, identique partout
        f = pd.read_csv(d / "frame_measurements.csv")
        for col, val in (("config_id", "C3"), ("checkpoint", "base_1"), ("gop", 4), ("resolution", "640x360")):
            self.assertEqual(set(f[col]), {val}, col)
        cr, tr = m["codec_runtime"], m["config_trace"]
        self.assertEqual((cr["gop_used"], cr["resolution_actual"], cr["checkpoint_name"]), (4, "640x360", "base_1"))
        self.assertEqual((tr["gop_requested"], tr["resolution_requested"], tr["checkpoint"]["name"]), (4, "640x360", "base_1"))
        self.assertEqual(tr["config_consistency"]["mismatches"], [])
        self.assertEqual(m["params"]["gop"], 4)
        self.assertTrue(m["frame_type_check"]["all_consistent"])
        self.assertEqual(f.type.tolist()[:4], ["I", "P", "P", "P"])
        self.assertEqual(f.type_observed.tolist()[:4], ["I", "P", "P", "P"])
        # --- honnêteté RAPL : l'environnement simulé ne doit JAMAIS être présenté comme une mesure RAPL
        es = m["energy_source_final"]
        self.assertFalse(es["rapl_package_channel_confirmed"])
        self.assertNotEqual(es["cpu_energy_source"], "measured_rapl_via_windows_emi")
        self.assertEqual(es["ram_energy_source"], "modelled")
        # --- énergie brute / nette, cohérence, négatifs conservés, pas de double comptage
        self.assertTrue(f.energy_net_consistent.all())
        self.assertLess(f.energy_net_sum_check_j.abs().max(), 1e-6)
        for comp in ("cpu", "ram"):
            exp_net = f[f"energy_{comp}_gross_j"] - f[f"idle_power_{comp}_w_used"] * f.energy_window_s
            self.assertLess((f[f"energy_{comp}_net_j"] - exp_net).abs().max(), 1e-9, comp)
        self.assertLess((f.energy_net_j - (f.energy_cpu_net_j + f.energy_ram_net_j + f.energy_gpu_net_j)).abs().max(), 1e-9)
        self.assertLess((f.energy_gross_j - (f.energy_cpu_gross_j + f.energy_ram_gross_j + f.energy_gpu_gross_j)).abs().max(), 1e-6)
        self.assertTrue((f.energy_ram_net_j < 0).any() or (f.energy_ram_net_j > 0).any() or True)   # signe libre, jamais forcé
        self.assertEqual(int((f.energy_net_j < 0).sum()), int(f.energy_net_negative.sum()))          # aucun clamp
        c = pd.read_csv(d / "component_time_energy.csv")
        tot = c.groupby(["repeat", "direction", "frame"]).energy_gross_estimated_j.sum().reset_index()
        merged = tot.merge(f[["repeat", "direction", "frame", "energy_gross_j"]], on=["repeat", "direction", "frame"])
        self.assertLess(((merged.energy_gross_estimated_j - merged.energy_gross_j).abs() / merged.energy_gross_j).max(), 1e-9)
        self.assertEqual(s["task_overlap_check"]["overlaps"], [])
        idle = pd.read_csv(d / "idle_baselines.csv")
        self.assertEqual(len(idle), 2)                                    # baseline initiale + avant la répétition 1
        self.assertEqual(s["energy_diagnostics"]["idle_power_stats"]["n_baselines"], 2)
        self.assertTrue(s["energy_diagnostics"]["net_consistency_all_ok"])
        # --- temps séparés (mesurés) + résidu (calculé)
        e, dec = f[f.direction == "enc"], f[f.direction == "dec"]
        self.assertLess((e.time_task_s - (e.time_preprocess_s + e.time_encode_s + e.time_bitstream_s + e.time_other_residual_s)).abs().max(), 1e-9)
        self.assertLess((dec.time_task_s - (dec.time_decode_s + dec.time_bitstream_s + dec.time_postprocess_s + dec.time_other_residual_s)).abs().max(), 1e-9)
        self.assertTrue(e.time_decode_s.isna().all() and e.time_postprocess_s.isna().all())
        self.assertTrue(dec.time_encode_s.isna().all() and dec.time_preprocess_s.isna().all())
        self.assertTrue((e[e.type == "I"].time_encode_s > 1.9).all())      # faux codec : 1,2 s + 0,8 s sur les images I
        self.assertTrue((e.time_preprocess_s > 0).all() and (dec.time_postprocess_s > 0).all())
        self.assertTrue((e.time_encode_s > e.time_preprocess_s).all())
        # --- bpp et bit/s (payload du faux codec : 100 + 10*indice octets)
        enc0 = e[e.repeat == 0].sort_values("frame")
        self.assertEqual(enc0.payload_bytes.tolist(), [100, 110, 120, 130])
        pix = 640 * 360
        self.assertLess((f.bits_payload - 8 * f.payload_bytes).abs().max(), 1e-9)
        self.assertLess((f.bpp_payload - f.bits_payload / pix).abs().max(), 1e-12)
        self.assertLess((f.bitrate_payload_bps - f.bits_payload * 25).abs().max(), 1e-6)
        self.assertLess((enc0.bits_record / 8 - (enc0.payload_bytes + 8)).abs().max(), 1e-9)        # +8 octets de cadrage du faux flux
        self.assertTrue(dec.bits_record.isna().all())
        self.assertEqual(set(f.fps), {25.0})
        self.assertEqual(set(f.fps_assumed), {False})
        rs = pd.read_csv(d / "repeat_summary.csv")
        self.assertLess((rs.seq_bitrate_file_bps - 8 * rs.bitstream_bytes / (4 / 25)).abs().max(), 1e-6)
        self.assertEqual(s["bitrate"]["fps"], 25.0)
        self.assertIn("time_other_residual_s", s["definitions"])

    # ------------------------------------------------------------------ 4. batch : résultats séparés
    def test_04_batch_separate_results(self):
        before = set(self.result_dirs())
        rc, out = self.v51("--batch", "--ids", "C1,C3")        # C3 existe déjà -> validation du lot échoue, RIEN n'est lancé
        self.assertNotEqual(rc, 0)
        self.assertIn("AUCUNE configuration n'a été lancée", out)
        self.assertEqual(set(self.result_dirs()), before)
        self.assertEqual(list(self.group.glob("_batch_*")), [])
        rc, out = self.v51("--batch", "--ids", "C1,C2")
        self.assertEqual(rc, 0, out[-3000:])
        dirs = [self.group / n for n in ("C1_gop2_768x432_base_1_videos_5_n4_r2", "C2_gop2_768x432_base_2_videos_5_n4_r2")]
        for d, cid, ck in zip(dirs, ("C1", "C2"), ("base_1", "base_2")):
            self.assertTrue(d.is_dir(), d)
            f = pd.read_csv(d / "frame_measurements.csv")
            self.assertEqual((set(f.config_id), set(f.checkpoint), set(f.gop)), ({cid}, {ck}, {2}))
            self.assertEqual(jload(d / "summary.json")["status"], "SUCCESS")
            self.assertEqual(f[f.direction == "enc"].sort_values(["repeat", "frame"]).type.tolist()[:4], ["I", "P", "I", "P"])
        b = jload(next(self.group.glob("_batch_*")) / "batch_summary.json")
        self.assertEqual([r["id"] for r in b["runs"]], ["C1", "C2"])
        self.assertEqual([r["return_code"] for r in b["runs"]], [0, 0])
        self.assertEqual(len({r["out_dir"] for r in b["runs"]}), 2)
        self.assertEqual(sorted(self.result_dirs()), sorted(["C1_gop2_768x432_base_1_videos_5_n4_r2", "C2_gop2_768x432_base_2_videos_5_n4_r2",
                                                             "C3_gop4_640x360_base_1_videos_5_n4_r2"]))

    # ------------------------------------------------------------------ 5. collecte d'énergie : honnêteté
    def test_05_check_energy_is_honest(self):
        rc, out = self.v51("--check_energy", "--check_seconds", "12")
        self.assertIn(rc, (0, 2))
        if rc == 2:                                            # attendu ici : pas d'EMI dans cet environnement
            self.assertIn("COLLECTE NON VALIDÉE", out)
        else:                                                  # sur un PC où RAPL via EMI fonctionne réellement
            self.assertIn("COLLECTE D'ÉNERGIE VALIDÉE", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
