# -*- coding: utf-8 -*-
"""Tests unitaires V5.1 (aucun matériel, aucun codec, aucune mesure réelle ; CodeCarbon et torch NON requis).
Lancer depuis svc_code :   python -m unittest test_energy_v51 -v
Ils vérifient la LOGIQUE (énergie, temps, bpp/bitrate, configuration). Ils ne valident NI RAPL NI le vrai codec."""
import ast
import copy
import math
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import energy_profiling_v51 as ep           # noqa: E402
import exp_config_v51 as xc                 # noqa: E402
import experiment3_frame_components_v51 as ex   # noqa: E402

REC = {"energy_total_j": 30.0, "energy_cpu_j": 25.0, "energy_ram_j": 5.0, "energy_gpu_j": 0.0, "energy_window_s": 4.0}
IDLE = {"power_total_w": 3.0, "power_cpu_w": 2.0, "power_ram_w": 1.0, "power_gpu_w": 0.0}


# ====================================================================== énergie (V4.1)
class TestEnergyBreakdown(unittest.TestCase):
    def test_gross_net_components_and_total(self):
        b = ep.energy_breakdown(REC, IDLE)
        self.assertAlmostEqual(b["net_cpu_j"], 25 - 2 * 4)
        self.assertAlmostEqual(b["net_ram_j"], 5 - 1 * 4)
        self.assertAlmostEqual(b["net_total_j"], b["net_cpu_j"] + b["net_ram_j"] + b["net_gpu_j"])
        self.assertAlmostEqual(b["net_total_j"], 30 - 3 * 4)
        self.assertTrue(b["consistent"])

    def test_negative_values_kept(self):
        b = ep.energy_breakdown(dict(REC, energy_cpu_j=1.0, energy_ram_j=4.0, energy_total_j=5.0), IDLE)
        self.assertAlmostEqual(b["net_cpu_j"], -7.0)
        self.assertAlmostEqual(b["net_total_j"], -7.0)
        self.assertTrue(b["consistent"])

    def test_inconsistency_detected(self):
        self.assertFalse(ep.energy_breakdown(dict(REC, energy_total_j=31.0), IDLE)["consistent"])
        self.assertFalse(ep.energy_breakdown(REC, dict(IDLE, power_total_w=3.5))["consistent"])

    def test_window_rule(self):
        rec = {k: v for k, v in REC.items() if k != "energy_window_s"}
        self.assertEqual(ep.energy_breakdown(dict(rec, cc_duration_s=5.0, wall_s=4.0), IDLE)["window_s"], 5.0)
        self.assertEqual(ep.energy_breakdown(dict(rec, cc_duration_s=float("nan"), wall_s=4.0), IDLE)["window_s"], 4.0)

    def test_idle_stats(self):
        rows = [{"power_cpu_w": 2.0, "power_ram_w": 1.0, "power_gpu_w": 0.0, "power_total_w": 3.0},
                {"power_cpu_w": 4.0, "power_ram_w": 1.0, "power_gpu_w": 0.0, "power_total_w": 5.0}]
        s = ep.idle_power_stats(rows)
        self.assertAlmostEqual(s["cpu"]["mean_w"], 3.0)
        self.assertAlmostEqual(s["cpu"]["std_w"], math.sqrt(2))
        self.assertEqual(ep.idle_power_stats([])["cpu"], {"n": 0})


class TestRaplClassification(unittest.TestCase):
    class Iface:
        def __init__(self, d):
            self._devices, self.emi_include_dram, self._mirrored_candidates = d, False, {}

    class CPU:
        def __init__(self, mode, iface):
            self._mode, self._intel_interface, self._tdp = mode, iface, 15

    class RAM:
        pass

    class Tr:
        def __init__(self, hw):
            self._hardware, self._conf = hw, {}

        def _ensure_hardware_ready(self):
            pass

        def get_detected_hardware(self):
            return {}

    ALL = ["RAPL_Package0_PKG", "RAPL_Package0_DRAM", "RAPL_Package0_PP0", "RAPL_Package0_PP1"]

    def meter(self, sel, n_tasks=2, nonpos=0, mode="windows_emi"):
        m = ep.EnergyMeter.__new__(ep.EnergyMeter)
        m.t = self.Tr([self.CPU(mode, self.Iface([("d", self.ALL, sel)])), self.RAM()])
        m.measure_power_secs, m.tracking_mode, m.country_iso_code = 1.0, "machine", "MAR"
        m.tasks, m.cpu_nonpositive_tasks = [{}] * n_tasks, ["x"] * nonpos
        return m

    def test_confirmed_only_with_package_and_active_counter(self):
        i = self.meter([0]).method_info()
        self.assertEqual(i["cpu_energy_source"], "measured_rapl_via_windows_emi")
        self.assertEqual(i["ram_energy_source"], "modelled")
        self.assertEqual(i["total_energy_kind"], "hybrid_cpu_measured_rapl_plus_ram_modelled")
        for sel in ([0, 1], [2], [0, 2], []):
            self.assertFalse(self.meter(sel).method_info()["cpu_is_confirmed_rapl_measurement"], sel)
        self.assertFalse(self.meter([0], nonpos=1).method_info()["cpu_is_confirmed_rapl_measurement"])
        self.assertEqual(self.meter([0], mode="cpu_load").method_info()["cpu_energy_source"], "modelled_tdp_x_load")


# ====================================================================== temps, bpp, bitrate (V4.1)
class TestTimeAndBitrate(unittest.TestCase):
    PIX = 768 * 432
    IO = {"image_read", "image_write", "bitstream_write", "bitstream_read", "preprocess_to_tensor", "other_io"}

    def test_io_stage_split(self):
        g = lambda n: ex.stage_group(n, self.IO)
        self.assertEqual(g("image_read")[0], "preprocess")
        self.assertEqual(g("preprocess_to_tensor")[0], "preprocess")
        self.assertEqual(g("bitstream_write")[0], "bitstream_io")
        self.assertEqual(g("bitstream_read")[0], "bitstream_io")
        self.assertEqual(g("image_write")[0], "postprocess")
        self.assertEqual(g("other_io")[0], "io")
        self.assertEqual(g("motion_estimation")[0], "motion_estimation")

    def test_keys_registered(self):
        for g in ("preprocess", "bitstream_io", "postprocess", "io", "uninstrumented"):
            self.assertIn(g, ex.GROUP_KEYS)
            self.assertIn(g, ex.GROUP_SHORT)
        for k in ex.TIME_SPLIT_KEYS + ex.BITRATE_KEYS:
            self.assertIn(k, ex.FRAME_KEYS)
            self.assertIn(k, ex.TYPE_KEYS)

    def test_bitrate_frame(self):
        m = ex.bitrate_metrics(1000, 1008, self.PIX, 30.0, True)
        self.assertEqual(m["bits_payload"], 8000.0)
        self.assertAlmostEqual(m["bpp_payload"], 8000 / self.PIX)
        self.assertAlmostEqual(m["bitrate_payload_bps"], 240000.0)
        self.assertAlmostEqual(m["bitrate_record_bps"], 8064 * 30)
        self.assertTrue(m["fps_assumed"])
        d = ex.bitrate_metrics(1000, float("nan"), self.PIX, 25.0)
        self.assertTrue(math.isnan(d["bitrate_record_bps"]))
        self.assertAlmostEqual(d["bitrate_payload_bps"], 200000.0)
        self.assertTrue(math.isnan(ex.bitrate_metrics(1000, 1008, self.PIX, 0.0)["bitrate_payload_bps"]))

    def test_bitrate_sequence(self):
        s = ex.sequence_bitrate(10000, 9000, 9500, 10, self.PIX, 25.0)
        self.assertAlmostEqual(s["sequence_duration_s"], 0.4)
        self.assertAlmostEqual(s["bitrate_file_bps"], 200000.0)
        self.assertAlmostEqual(s["bpp_file_mean"], 8 * 10000 / (10 * self.PIX))

    def test_observed_frame_type_without_stage_name_assumptions(self):
        class P:
            records = ([{"dir": "enc", "frame": 0, "phase": "measured", "stage": s} for s in ("a", "b")] +
                       [{"dir": "enc", "frame": 1, "phase": "measured", "stage": s} for s in ("a", "c")] +
                       [{"dir": "enc", "frame": 2, "phase": "measured", "stage": s} for s in ("b", "a")])
        self.assertEqual(ex.observed_frame_type(P, "enc", 0), "I")
        self.assertEqual(ex.observed_frame_type(P, "enc", 1), "P")
        self.assertEqual(ex.observed_frame_type(P, "enc", 2), "I")
        self.assertEqual(ex.observed_frame_type(P, "dec", 1), "?")


# ====================================================================== configuration (V5)
def make_project(tmp, n_src=6, size=(768, 432)):
    tmp = Path(tmp)
    (tmp / "videos_5" / "input_png").mkdir(parents=True)
    (tmp / "checkpoints" / "base").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(1, n_src + 1):
        Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(tmp / "videos_5" / "input_png" / f"frame_{i:03d}.png")
    for k in (1, 2, 3, 4):
        (tmp / "checkpoints" / "base" / f"checkpoint_base_{k}.pth.tar").write_bytes(b"x%d" % k)
    cfg = yaml.safe_load((HERE / "experiment_config_v51.yaml").read_text(encoding="utf-8"))
    cfg["paths"]["results_root"] = str(tmp / "results")
    cfg["defaults"]["frames"]["count"] = 4
    cfg["defaults"]["repeats"] = 2
    cfg["defaults"]["video"]["fps"] = 25
    cfg["defaults"]["energy"].update(measure_power_secs=0.2, idle_seconds=3, idle_between_seconds=2, cooldown_seconds=1)
    p = tmp / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return tmp, cfg, p


def status(checks, name_part):
    return [c["status"] for c in checks if name_part in c["check"]]


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.root, self.cfg, self.path = make_project(self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def load(self, mutate=None):
        cfg = copy.deepcopy(self.cfg)
        if mutate:
            mutate(cfg)
        self.path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return xc.load_config(self.path)

    def test_shipped_yaml_keeps_c1_to_c4(self):
        c = yaml.safe_load((HERE / "experiment_config_v51.yaml").read_text(encoding="utf-8"))
        got = {e["id"]: (e["gop"], e["resolution"], e["checkpoint"]) for e in c["experiments"]}
        self.assertEqual(got, {"C1": (4, "768x432", "base_1"), "C2": (4, "768x432", "base_2"),
                               "C3": (4, "640x360", "base_1"), "C4": (8, "768x432", "base_1")})
        self.assertEqual(c["allowed"]["gop"][:4], [2, 4, 8, 16])
        self.assertEqual(c["allowed"]["resolution"], ["768x432", "640x360", "384x216"])
        self.assertEqual(c["allowed"]["checkpoint"], ["base_1", "base_2", "base_3", "base_4"])
        self.assertEqual(c["paths"]["results_group"], "exp3_v51")
        self.assertIn("detector", c["defaults"])

    def test_resolve_and_naming(self):
        cfg = self.load()
        R = xc.resolve(cfg, "C3", self.root)
        self.assertEqual((R["gop"], R["W"], R["H"], R["checkpoint"], R["ckpt_id"]), (4, 640, 360, "base_1", 1))
        self.assertEqual(R["paths"]["dirname"], "C3_gop4_640x360_base_1_videos_5_n4_r2")
        self.assertTrue(R["paths"]["out_dir"].endswith(os.path.join("exp3_v51", R["paths"]["dirname"])))
        self.assertEqual(R["fps"], 25.0)
        self.assertFalse(R["fps_assumed"])
        names = {xc.resolve(cfg, i, self.root)["paths"]["dirname"] for i in xc.list_ids(cfg)}
        self.assertEqual(len(names), 4)                               # un dossier distinct par configuration

    def test_valid_config_has_no_failure(self):
        cfg = self.load()
        for i in ("C1", "C2", "C3", "C4"):
            ch = xc.static_checks(xc.resolve(cfg, i, self.root))
            self.assertFalse(xc.has_fail(ch), [c for c in ch if c["status"] == xc.FAIL])
        self.assertEqual(status(xc.static_checks(xc.resolve(cfg, "C1", self.root)), "structure I/P attendue"), [xc.OK])

    def test_invalid_gop_resolution_checkpoint(self):
        def mut(c):
            c["experiments"] += [{"id": "X1", "gop": 3}, {"id": "X2", "resolution": "800x600"}, {"id": "X3", "checkpoint": "base_9"}]
        cfg = self.load(mut)
        self.assertIn(xc.FAIL, status(xc.static_checks(xc.resolve(cfg, "X1", self.root)), "valeur autorisée : gop"))
        self.assertIn(xc.FAIL, status(xc.static_checks(xc.resolve(cfg, "X2", self.root)), "valeur autorisée : resolution"))
        ch = xc.static_checks(xc.resolve(cfg, "X3", self.root))
        self.assertIn(xc.FAIL, status(ch, "valeur autorisée : checkpoint"))
        self.assertIn(xc.FAIL, status(ch, "checkpoint : fichier"))
        with self.assertRaises(xc.ConfigError):
            xc.parse_resolution("768-432")
        with self.assertRaises(xc.ConfigError):
            xc.resolve(cfg, "C99", self.root)

    def test_frames_not_available(self):
        cfg = self.load(lambda c: c["experiments"].append({"id": "X4", "frames": {"count": 50}}))
        self.assertIn(xc.FAIL, status(xc.static_checks(xc.resolve(cfg, "X4", self.root)), "images disponibles"))

    def test_gop_structure_expected(self):
        self.assertEqual("".join(xc.expected_types(10, 4)), "IPPPIPPPIP")
        self.assertEqual("".join(xc.expected_types(10, 2)), "IPIPIPIPIP")
        self.assertEqual("".join(xc.expected_types(10, 16)), "IPPPPPPPPP")

    def test_resolution_plan_and_resize(self):
        cfg = self.load()
        R = xc.resolve(cfg, "C3", self.root)
        plan = xc.input_plan(R)
        self.assertEqual(plan["action"], "resize")
        info = xc.prepare_inputs(R, plan)
        self.assertEqual(len(info["created"]), 4)
        for i in range(4):
            with Image.open(xc.frame_path(plan["inputs_dir"], "frame_", 1 + i)) as im:
                self.assertEqual(im.size, (640, 360))
        with Image.open(xc.frame_path(R["paths"]["frames_dir"], "frame_", 1)) as im:        # source intacte
            self.assertEqual(im.size, (768, 432))
        self.assertEqual(len(xc.prepare_inputs(R, plan)["reused"]), 4)                       # jamais écrasé
        bad = Path(xc.frame_path(plan["inputs_dir"], "frame_", 1))
        Image.new("RGB", (10, 10)).save(bad)
        with self.assertRaises(xc.ConfigError):
            xc.prepare_inputs(R, plan)
        self.assertEqual(xc.input_plan(xc.resolve(cfg, "C1", self.root))["action"], "use_source")

    def test_aspect_and_upscale_guards(self):
        def mut(c):
            c["allowed"]["resolution"].append("640x480")
            c["experiments"].append({"id": "X5", "resolution": "640x480"})
        cfg = self.load(mut)
        self.assertIn(xc.FAIL, status(xc.static_checks(xc.resolve(cfg, "X5", self.root)), "rapport d'aspect"))
        shutil.rmtree(self.root / "videos_5")
        make_project_small = Path(self.td) / "videos_5" / "input_png"
        make_project_small.mkdir(parents=True)
        for i in range(1, 7):
            Image.new("RGB", (384, 216)).save(make_project_small / f"frame_{i:03d}.png")
        self.assertIn(xc.FAIL, status(xc.static_checks(xc.resolve(self.load(), "C3", self.root)), "agrandissement"))

    def test_detector_must_be_identical_across_batch(self):
        cfg = self.load(lambda c: c["experiments"].append({"id": "X6", "detector": {"conf_threshold": 0.5}}))
        Rs = [xc.resolve(cfg, i, self.root) for i in ("C1", "X6")]
        self.assertIn(xc.FAIL, status(xc.cross_checks(Rs), "détecteur identique"))
        Rs = [xc.resolve(cfg, i, self.root) for i in ("C1", "C2", "C3")]
        self.assertNotIn(xc.FAIL, status(xc.cross_checks(Rs), "détecteur identique"))

    def test_energy_and_fps_rules(self):
        cfg = self.load(lambda c: c["experiments"].append({"id": "X7", "energy": {"idle_seconds": 1.0}}))
        self.assertTrue(xc.has_fail(xc.static_checks(xc.resolve(cfg, "X7", self.root))))
        cfg = self.load(lambda c: c["defaults"]["video"].update(fps=None))
        R = xc.resolve(cfg, "C1", self.root)
        self.assertTrue(R["fps_assumed"] and R["fps"] == 30.0)
        self.assertEqual(status(xc.static_checks(R), "cadence fps"), [xc.WARN])
        cfg = self.load(lambda c: c["defaults"]["video"].update(fps=-5))
        self.assertIn(xc.FAIL, status(xc.static_checks(xc.resolve(cfg, "C1", self.root)), "cadence fps"))

    def test_existing_output_dir_is_refused(self):
        cfg = self.load()
        R = xc.resolve(cfg, "C1", self.root)
        Path(R["paths"]["out_dir"]).mkdir(parents=True)
        (Path(R["paths"]["out_dir"]) / "old.txt").write_text("previous results")
        self.assertIn(xc.FAIL, status(xc.static_checks(R), "dossier de résultats"))

    def test_single_configuration_namespace_consistency(self):
        R = xc.resolve(self.load(), "C3", self.root)
        a = xc.to_namespace(R, R["paths"]["inputs_cache"])
        self.assertEqual(xc.namespace_consistency(R, a), [])
        self.assertEqual((a.gop, a.width, a.height, a.checkpoint, a.fps), (4, 640, 360, 1, 25.0))
        a.gop = 8
        self.assertTrue(any(m.startswith("gop") for m in xc.namespace_consistency(R, a)))

    def test_checkpoint_comparison_does_not_assume_quality(self):
        reps = {"base_1": {"signature_sha256": "A", "config_like_fields": {"lmbda": 0.01}},
                "base_2": {"signature_sha256": "A", "config_like_fields": {"lmbda": 0.02}},
                "base_3": {"signature_sha256": "B", "config_like_fields": {"lmbda": 0.02}}}
        c = xc.compare_checkpoints(reps)
        self.assertFalse(c["same_signature"])
        self.assertIn("lmbda", c["differing_config_fields"])
        cross = xc.cross_checks([xc.resolve(self.load(), "C1", self.root)], deep=True, ckpt_cache=reps)
        self.assertIn(xc.WARN, status(cross, "mêmes noms et formes"))           # avertissement, pas un échec

    def test_estimate_scales_with_pixels(self):
        cfg = self.load()
        e1, f1 = xc.estimate_params(xc.resolve(cfg, "C1", self.root))
        e3, f3 = xc.estimate_params(xc.resolve(cfg, "C3", self.root))
        self.assertAlmostEqual(f1, 1.0)
        self.assertAlmostEqual(f3, (640 * 360) / (768 * 432))


# ====================================================================== imports / dépendances
class TestImportsAndDependencies(unittest.TestCase):
    THIRD_PARTY = {"yaml", "PIL", "numpy", "matplotlib", "psutil", "codecarbon", "torch", "torchvision", "constriction"}
    LOCAL = {"exp_config_v51", "energy_profiling_v51", "test_base_profiled", "base_codec", "bitstream", "profiling",
             "entropy_codec", "experiment3_frame_components_v51"}

    def roots(self, path, top_level_only=False):
        out = set()
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        for n in (tree.body if top_level_only else ast.walk(tree)):
            if isinstance(n, ast.Import):
                out |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                out.add(n.module.split(".")[0])
        return out

    def test_every_import_is_stdlib_known_third_party_or_project_module(self):
        std = set(sys.stdlib_module_names)
        for f in ("experiment3_frame_components_v51.py", "exp_config_v51.py", "energy_profiling_v51.py"):
            unknown = {r for r in self.roots(HERE / f)} - std - self.THIRD_PARTY - self.LOCAL - {"__future__"}
            self.assertFalse(unknown, f"{f}: imports inconnus {unknown}")

    def test_no_reference_to_original_modules(self):
        stale = {"energy_profiling", "exp_config", "energy_profiling_v41", "experiment3_frame_components_v4",
                 "experiment3_frame_components_v41", "experiment3_frame_components_v5"}
        for f in ("experiment3_frame_components_v51.py", "exp_config_v51.py", "energy_profiling_v51.py"):
            self.assertFalse(self.roots(HERE / f) & stale, f"{f} importe encore un module d'une version précédente")
            txt = (HERE / f).read_text(encoding="utf-8")
            for old in ("import energy_profiling as", "from energy_profiling import", "import exp_config as"):
                self.assertNotIn(old, txt, f"{f} référence encore {old}")
        self.assertIn("energy_profiling_v51.py", ex.REQUIRED)
        self.assertEqual(ex.VERSION, "v51")

    def test_light_modules_importable_without_heavy_dependencies(self):
        for m in ("torch", "torchvision", "codecarbon"):
            self.assertNotIn(m, self.roots(HERE / "exp_config_v51.py", top_level_only=True))     # imports paresseux seulement


if __name__ == "__main__":
    unittest.main(verbosity=2)
