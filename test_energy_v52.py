# -*- coding: utf-8 -*-
"""
test_energy_v52.py -- tests du module énergétique autonome V5.2. N'utilise NI le codec VCM, NI les checkpoints, NI YOLO,
NI CodeCarbon (sauf les tests matériels, désactivés par défaut).

Quatre catégories, distinguées par le NOM DE LA CLASSE et rappelées dans le bilan final :
  Unit_*       tests unitaires de fonctions pures ou de fichiers (aucune mesure, aucun backend)
  Simulated_*  tests avec backend SIMULÉ (FakeBackend) : valident la plomberie, JAMAIS l'énergie matérielle
  Hardware_*   tests matériels Windows : exécutés SEULEMENT si V52_HARDWARE=1 sous Windows avec codecarbon installé ;
               sinon ils sont comptés "NON EXÉCUTÉS"
Lancer :  python test_energy_v52.py        (ou  python -m unittest test_energy_v52 -v)
"""
import ast
import copy
import dataclasses
import json
import math
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import energy_profiling_v52 as ep      # noqa: E402
import exp_config_v52 as xc            # noqa: E402

NAN = float("nan")
FAST = ep.MeterConfig(measure_power_secs=0.05, min_window_periods=4)      # fenêtre minimale = 0,2 s (tests rapides)


def idle_rec(cpu, ram=1.0, gpu=0.0):
    return {"mean_power_cpu_w": cpu, "mean_power_ram_modelled_w": ram, "mean_power_gpu_w": gpu,
            "mean_power_total_hybrid_w": cpu + ram + gpu}


def task_rec(cpu_j, ram_j, win, gpu_j=0.0):
    return {"window_s": win, "energy_cpu_j": cpu_j, "energy_ram_modelled_j": ram_j, "energy_gpu_j": gpu_j,
            "energy_total_hybrid_j": cpu_j + ram_j + gpu_j}


# ======================================================================================================
class Unit_EnergyMaths(unittest.TestCase):
    def test_gross_and_net(self):
        base = ep.idle_baseline([idle_rec(2.0), idle_rec(2.0)])
        e = ep.compute_energy(task_rec(25.0, 5.0, 4.0), base, sampling_period_s=1.0)
        self.assertAlmostEqual(e["net_cpu_j"], 25 - 2 * 4)
        self.assertAlmostEqual(e["net_ram_modelled_j"], 5 - 1 * 4)
        self.assertAlmostEqual(e["net_total_hybrid_j"], 30 - 3 * 4)
        self.assertAlmostEqual(e["net_component_sum_j"], e["net_total_hybrid_j"])
        self.assertAlmostEqual(e["idle_energy_cpu_j"], 8.0)
        self.assertTrue(e["net_consistent"])

    def test_inconsistent_total_detected(self):
        base = ep.idle_baseline([idle_rec(2.0), idle_rec(2.0)])
        rec = task_rec(25.0, 5.0, 4.0)
        rec["energy_total_hybrid_j"] = 31.0
        self.assertFalse(ep.compute_energy(rec, base, sampling_period_s=1.0)["net_consistent"])

    def test_mean_power(self):
        self.assertAlmostEqual(ep.mean_power_w(30.0, 4.0), 7.5)
        for bad in ((30.0, 0.0), (30.0, -1.0), (NAN, 4.0), (30.0, NAN), (30.0, float("inf"))):
            self.assertTrue(math.isnan(ep.mean_power_w(*bad)), bad)

    def test_window_choice_and_gap(self):
        self.assertEqual(ep.choose_window(5.0, 4.0), (5.0, "backend_duration_s"))
        self.assertEqual(ep.choose_window(NAN, 4.0), (4.0, "wall_s_fallback"))
        self.assertEqual(ep.choose_window(0.0, NAN)[1], "invalid")
        gap_s, gap_pct = ep.window_gap(4.2, 4.0)
        self.assertAlmostEqual(gap_s, 0.2)
        self.assertAlmostEqual(gap_pct, 5.0)

    def test_uncertainty_values(self):
        # idle CPU 2 W et 4 W : moyenne 3, écart-type sqrt(2), erreur-type 1 ; fenêtre 10 s -> u_idle = 10 J
        base = ep.idle_baseline([idle_rec(2.0), idle_rec(4.0)])
        self.assertAlmostEqual(base["cpu"]["mean_w"], 3.0)
        self.assertAlmostEqual(base["cpu"]["sd_w"], math.sqrt(2))
        self.assertAlmostEqual(base["cpu"]["sem_w"], 1.0)
        e = ep.compute_energy(task_rec(100.0, 10.0, 10.0), base, sampling_period_s=1.0, boundary_half_width_periods=1.0)
        self.assertAlmostEqual(e["u_idle_cpu_j"], 10.0)
        u_b = 1.0 * 1.0 * (100.0 / 10.0) / math.sqrt(3)                 # h x période x P_tâche / sqrt(3)
        self.assertAlmostEqual(e["u_boundary_cpu_j"], u_b)
        self.assertAlmostEqual(e["u_net_cpu_j"], math.sqrt(100.0 + u_b ** 2))
        self.assertAlmostEqual(e["net_cpu_j"], 100 - 30)
        self.assertTrue(e["net_cpu_significant"])
        self.assertAlmostEqual(e["u_net_cpu_j_per_item"], e["u_net_cpu_j"])

    def test_uncertainty_undefined_with_single_idle(self):
        e = ep.compute_energy(task_rec(100.0, 10.0, 10.0), ep.idle_baseline([idle_rec(3.0)]), sampling_period_s=1.0)
        self.assertTrue(math.isnan(e["u_net_cpu_j"]))
        self.assertIsNone(e["net_cpu_significant"])
        self.assertAlmostEqual(e["net_cpu_j"], 70.0)                      # le net reste calculé

    def test_boundary_model_can_be_disabled(self):
        base = ep.idle_baseline([idle_rec(2.0), idle_rec(4.0)])
        e = ep.compute_energy(task_rec(100.0, 10.0, 10.0), base, sampling_period_s=1.0, boundary_enabled=False)
        self.assertEqual(e["u_boundary_cpu_j"], 0.0)
        self.assertAlmostEqual(e["u_net_cpu_j"], 10.0)

    def test_negative_values_kept(self):
        base = ep.idle_baseline([idle_rec(5.0), idle_rec(5.0)])
        e = ep.compute_energy(task_rec(10.0, 4.0, 4.0), base, sampling_period_s=1.0)   # 10 - 5x4 = -10
        self.assertAlmostEqual(e["net_cpu_j"], -10.0)
        self.assertTrue(e["net_cpu_negative"])
        self.assertLess(e["net_total_hybrid_j"], 0)
        self.assertTrue(e["net_consistent"])                              # jamais de clamp ni de valeur absolue

    def test_negative_within_or_beyond_uncertainty(self):
        base = ep.idle_baseline([idle_rec(2.0), idle_rec(4.0)])           # u_idle_cpu = sem x fenêtre
        within = ep.compute_energy(task_rec(28.0, 3.0, 10.0), base, sampling_period_s=1.0)   # net = -2, u_net >= 10
        self.assertTrue(within["net_cpu_negative"])
        self.assertTrue(within["net_cpu_negative_within_uncertainty"])
        base2 = ep.idle_baseline([idle_rec(5.0), idle_rec(5.0)])          # sem = 0 -> u = bord seul (faible)
        beyond = ep.compute_energy(task_rec(1.0, 1.0, 10.0), base2, sampling_period_s=0.001)
        self.assertTrue(beyond["net_cpu_negative"])
        self.assertFalse(beyond["net_cpu_negative_within_uncertainty"])

    def test_per_item_scaling(self):
        base = ep.idle_baseline([idle_rec(2.0), idle_rec(2.0)])
        e = ep.compute_energy(task_rec(50.0, 10.0, 10.0), base, sampling_period_s=1.0, n_items=20)
        self.assertAlmostEqual(e["energy_cpu_j_per_item"], 2.5)
        self.assertAlmostEqual(e["net_cpu_j_per_item"], (50 - 20) / 20)

    def test_idle_drift(self):
        d = ep.idle_drift(idle_rec(3.0), idle_rec(3.6))
        self.assertAlmostEqual(d["cpu"], 0.2)


class Unit_Validation(unittest.TestCase):
    def rec(self, **kw):
        r = {"window_s": 1.0, "window_gap_pct": 0.5, "t_start_mono_ns": 1, "t_stop_mono_ns": 10, "t_start_wall_ns": 1,
             "t_stop_wall_ns": 10, "energy_cpu_j": 5.0, "energy_ram_modelled_j": 1.0, "energy_gpu_j": 0.0,
             "energy_total_hybrid_j": 6.0, "error": None}
        r.update(kw)
        return r

    def status(self, checks, name):
        return [c["status"] for c in checks if c["check"] == name]

    def test_short_window_detected(self):
        cfg = ep.MeterConfig(measure_power_secs=1.0, min_window_periods=10)
        self.assertEqual(self.status(ep.validate_task(self.rec(window_s=3.0), cfg), "fenetre_ge_periodes"), ["fail"])
        self.assertEqual(self.status(ep.validate_task(self.rec(window_s=10.0), cfg), "fenetre_ge_periodes"), ["ok"])
        self.assertEqual(self.status(ep.validate_task(self.rec(window_s=0.05), cfg), "fenetre_ge_periodes"), ["fail"])

    def test_duration_consistency(self):
        cfg = ep.MeterConfig(duration_tolerance_pct=2.0)
        self.assertEqual(self.status(ep.validate_task(self.rec(window_gap_pct=1.0), cfg), "coherence_durees"), ["ok"])
        self.assertEqual(self.status(ep.validate_task(self.rec(window_gap_pct=5.0), cfg), "coherence_durees"), ["warn"])
        self.assertEqual(self.status(ep.validate_task(self.rec(window_gap_pct=NAN), cfg), "coherence_durees"), ["fail"])

    def test_timestamps_and_nonfinite(self):
        cfg = ep.MeterConfig()
        self.assertEqual(self.status(ep.validate_task(self.rec(t_stop_mono_ns=0), cfg), "horodatages_ordonnes"), ["fail"])
        self.assertEqual(self.status(ep.validate_task(self.rec(energy_cpu_j=NAN), cfg), "energies_finies"), ["fail"])
        self.assertTrue(ep.has_fail(ep.validate_task(self.rec(window_s=NAN), cfg)))

    def test_total_equals_sum(self):
        cfg = ep.MeterConfig()
        self.assertEqual(self.status(ep.validate_task(self.rec(energy_total_hybrid_j=9.0), cfg), "total_egal_somme_composants"), ["fail"])

    def test_cpu_nonpositive_warns(self):
        self.assertEqual(self.status(ep.validate_task(self.rec(energy_cpu_j=0.0, energy_total_hybrid_j=1.0), ep.MeterConfig()),
                                     "energie_cpu_positive"), ["warn"])

    def test_load_idle_ratio_is_configurable_control(self):
        base = ep.idle_baseline([idle_rec(5.0), idle_rec(5.0)])
        low = {"mean_power_cpu_w": 5.5, "mean_power_total_hybrid_w": 6.5}       # rapport 1,1 < 1,2
        warn = ep.validate_load_vs_idle(low, base, ep.MeterConfig())
        self.assertEqual([c["status"] for c in warn if c["check"] == "charge_sur_idle_cpu"], ["warn"])
        self.assertAlmostEqual([c["value"] for c in warn if c["check"] == "charge_sur_idle_cpu"][0], 1.1)
        strict = ep.validate_load_vs_idle(low, base, ep.MeterConfig(load_idle_ratio_severity="fail"))
        self.assertTrue(ep.has_fail(strict))
        off = ep.validate_load_vs_idle(low, base, ep.MeterConfig(load_idle_ratio_check=False))
        self.assertEqual([c["status"] for c in off if c["check"] == "charge_sur_idle_cpu"], ["info"])
        self.assertAlmostEqual([c["value"] for c in off if c["check"] == "charge_sur_idle_cpu"][0], 1.1)   # valeur toujours enregistrée
        looser = ep.validate_load_vs_idle(low, base, ep.MeterConfig(load_idle_ratio_min=1.05))
        self.assertEqual([c["status"] for c in looser if c["check"] == "charge_sur_idle_cpu"], ["ok"])

    def test_idle_validation(self):
        base = ep.idle_baseline([idle_rec(3.0), idle_rec(4.0)])
        checks = ep.validate_idle(base, ep.MeterConfig(), ep.idle_drift(idle_rec(3.0), idle_rec(4.0)))
        self.assertIn("warn", [c["status"] for c in checks if c["check"] == "idle_cv_cpu"])
        self.assertIn("warn", [c["status"] for c in checks if c["check"] == "idle_derive_cpu"])
        one = ep.validate_idle(ep.idle_baseline([idle_rec(3.0)]), ep.MeterConfig())
        self.assertIn("warn", [c["status"] for c in one if c["check"] == "idle_nombre_mesures"])


class Unit_Sources(unittest.TestCase):
    ALL = ["RAPL_Package0_PKG", "RAPL_Package0_DRAM", "RAPL_Package0_PP0", "RAPL_Package0_PP1"]

    def probe(self, names, ok=True):
        return {"probe_ok": ok, "selected_channel_names": names, "include_dram": False}

    def cls(self, names, mode="windows_emi", n=2, nonpos=0, **kw):
        return ep.classify_sources(mode, self.probe(names), n, nonpos, "RAPL_Package0_PKG", **kw)

    def test_confirmed_cpu_measured_ram_modelled_total_hybrid(self):
        c = self.cls(["RAPL_Package0_PKG"])
        self.assertEqual(c["cpu_energy_source"], "measured_rapl_via_windows_emi")
        self.assertTrue(c["cpu_is_confirmed_rapl_measurement"])
        self.assertEqual(c["ram_energy_source"], "modelled")
        self.assertEqual(c["total_energy_kind"], "hybrid_cpu_measured_rapl_plus_ram_modelled")
        self.assertEqual(c["primary_energy_metric"], "cpu_package_rapl")

    def test_not_confirmed_cases(self):
        for names in (["RAPL_Package0_PKG", "RAPL_Package0_DRAM"], ["RAPL_Package0_PP0"], ["RAPL_Package0_DRAM"], [],
                      ["RAPL_Package1_PKG"]):
            c = self.cls(names)
            self.assertFalse(c["cpu_is_confirmed_rapl_measurement"], names)
            self.assertEqual(c["ram_energy_source"], "modelled")
            self.assertEqual(c["primary_energy_metric"], "none_confirmed")
        self.assertFalse(self.cls(["RAPL_Package0_PKG"], nonpos=1)["cpu_is_confirmed_rapl_measurement"])
        self.assertFalse(ep.classify_sources("windows_emi", self.probe(["RAPL_Package0_PKG"], ok=False), 2, 0, "")
                         ["cpu_is_confirmed_rapl_measurement"])

    def test_before_first_task_not_confirmed(self):
        c = self.cls(["RAPL_Package0_PKG"], n=0)
        self.assertEqual(c["cpu_energy_source"], "windows_emi_package_channel_confirmed_energy_not_yet_checked")
        self.assertFalse(c["cpu_is_confirmed_rapl_measurement"])

    def test_modelled_and_simulated_modes(self):
        self.assertEqual(self.cls(["x"], mode="cpu_load")["cpu_energy_source"], "modelled_tdp_x_load")
        self.assertEqual(self.cls(["x"], mode="constant")["cpu_energy_source"], "modelled_constant_tdp")
        self.assertEqual(self.cls(["x"], mode="intel_rapl")["cpu_energy_source"],
                         "intel_rapl_powercap_domain_not_verified_by_this_module")
        s = self.cls(["RAPL_Package0_PKG"], simulated=True)
        self.assertEqual(s["cpu_energy_source"], "simulated_not_hardware")
        self.assertFalse(s["cpu_is_confirmed_rapl_measurement"])      # un résultat simulé n'est JAMAIS "mesuré"
        self.assertEqual(s["total_energy_kind"], "simulated_not_a_measurement")

    def test_package_channel_names(self):
        for n, exp in (("RAPL_Package0_PKG", True), ("RAPL_Package0_DRAM", False), ("RAPL_Package0_PP0", False),
                       ("RAPL_Package0_PP1", False), ("RAPL_Psys", False)):
            self.assertEqual(ep.is_package_channel_name(n), exp, n)


class Unit_StrictJson(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_nonfinite_become_null(self):
        obj = {"a": NAN, "b": float("inf"), "c": -float("inf"), "d": [1.0, NAN, {"e": float("inf")}], "f": (NAN,), "ok": 1.5}
        s = ep.dumps_strict(obj)
        self.assertNotIn("NaN", s)
        self.assertNotIn("Infinity", s)
        back = ep.loads_strict(s)
        self.assertEqual(back, {"a": None, "b": None, "c": None, "d": [1.0, None, {"e": None}], "f": [None], "ok": 1.5})

    def test_strict_parser_rejects_nonstandard(self):
        for bad in ('{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}'):
            with self.assertRaises(ValueError):
                ep.loads_strict(bad)

    def test_special_types(self):
        @dataclasses.dataclass
        class D:
            x: float
            p: Path

        s = ep.dumps_strict({"d": D(NAN, Path("a/b")), "set": {3, 1, 2}, 5: "int key"})
        self.assertEqual(ep.loads_strict(s), {"d": {"x": None, "p": str(Path("a/b"))}, "set": [1, 2, 3], "5": "int key"})

    def test_file_write_is_strict_and_refuses_overwrite(self):
        p = self.td / "m.json"
        ep.dump_json_strict(p, {"v": NAN, "w": float("inf")})
        self.assertEqual(ep.loads_strict(p.read_text(encoding="utf-8")), {"v": None, "w": None})
        with self.assertRaises(FileExistsError):
            ep.dump_json_strict(p, {"v": 1})
        self.assertEqual(ep.loads_strict(p.read_text(encoding="utf-8")), {"v": None, "w": None})   # inchangé
        ep.dump_json_strict(p, {"v": 2}, overwrite=True)
        self.assertEqual(ep.loads_strict(p.read_text(encoding="utf-8")), {"v": 2})

    def test_csv_cell(self):
        self.assertEqual(ep.csv_cell(NAN), "")
        self.assertEqual(ep.csv_cell(float("inf")), "")
        self.assertEqual(ep.csv_cell(1.5), 1.5)
        self.assertEqual(ep.csv_cell("x"), "x")

    def test_rawlog_refuses_existing_file(self):
        p = self.td / "raw.jsonl"
        r = ep.RawLog(p)
        r.append({"v": NAN})
        r.close()
        self.assertEqual(ep.loads_strict(p.read_text(encoding="utf-8").strip()), {"v": None})
        with self.assertRaises(FileExistsError):
            ep.RawLog(p)


class Unit_Isolation(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_creates_only_in_v52_group_and_refuses_existing(self):
        p = ep.resolve_output_dir(self.td / "results" / "VCM_EnergyMap", "exp3_v52", "run1")
        self.assertTrue(p.is_dir())
        self.assertEqual(p.parent.name, "exp3_v52")
        (p / "keep.txt").write_text("x")
        with self.assertRaises(FileExistsError):
            ep.resolve_output_dir(self.td / "results" / "VCM_EnergyMap", "exp3_v52", "run1")
        self.assertTrue((p / "keep.txt").exists())                        # rien supprimé ni écrasé

    def test_refuses_other_groups_and_unsafe_names(self):
        root = self.td / "r"
        for group in ("exp3_v51", "exp3_v5", "exp3_v41", "exp3", ""):
            with self.assertRaises(ep.IsolationError, msg=group):
                ep.resolve_output_dir(root, group, "run")
        for name in ("..", ".", "a/b", "a\\b", "", "../exp3_v51/x", "run exp", "exp3_v51_copy"):
            with self.assertRaises(ep.IsolationError, msg=name):
                ep.resolve_output_dir(root, "exp3_v52", name)
        self.assertFalse(root.exists())                                    # aucun dossier créé en cas de refus

    def test_v51_directory_never_touched(self):
        v51 = self.td / "results" / "VCM_EnergyMap" / "exp3_v51" / "C1"
        v51.mkdir(parents=True)
        (v51 / "result.csv").write_text("v51")
        ep.resolve_output_dir(self.td / "results" / "VCM_EnergyMap", "exp3_v52", "C1")
        self.assertEqual((v51 / "result.csv").read_text(), "v51")
        self.assertEqual(sorted(p.name for p in (self.td / "results" / "VCM_EnergyMap").iterdir()), ["exp3_v51", "exp3_v52"])

    def test_constants_agree_across_modules(self):
        self.assertEqual(ep.REQUIRED_RESULTS_GROUP, xc.REQUIRED_RESULTS_GROUP)
        self.assertEqual(ep.REQUIRED_RESULTS_GROUP, "exp3_v52")


class Unit_Independence(unittest.TestCase):
    """V5.2 n'importe rien de V5.1, du codec VCM, de torch/YOLO ; CodeCarbon seulement en import paresseux."""
    FORBIDDEN = {"energy_profiling_v51", "exp_config_v51", "experiment3_frame_components_v51", "energy_profiling_v41",
                 "energy_profiling", "exp_config", "base_codec", "bitstream", "entropy_codec", "profiling",
                 "test_base_profiled", "torch", "torchvision", "ultralytics"}

    def roots(self, f, top_only):
        tree = ast.parse((HERE / f).read_text(encoding="utf-8"))
        out = set()
        for n in (tree.body if top_only else ast.walk(tree)):
            if isinstance(n, ast.Import):
                out |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                out.add(n.module.split(".")[0])
        return out

    def test_no_forbidden_imports_anywhere(self):
        for f in ("energy_profiling_v52.py", "exp_config_v52.py"):
            self.assertFalse(self.roots(f, False) & self.FORBIDDEN, f)

    def test_codecarbon_yaml_psutil_not_imported_at_top_level(self):
        for f in ("energy_profiling_v52.py", "exp_config_v52.py"):
            self.assertFalse(self.roots(f, True) & {"codecarbon", "yaml", "psutil"}, f)

    def test_energy_module_loads_without_codecarbon(self):
        self.assertNotIn("codecarbon", sys.modules)


class Unit_Config(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        import yaml
        self.yaml = yaml
        self.base = yaml.safe_load((HERE / "experiment_config_v52.yaml").read_text(encoding="utf-8"))

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def load(self, mutate=None):
        d = copy.deepcopy(self.base)
        if mutate:
            mutate(d)
        p = self.td / "c.yaml"
        p.write_text(self.yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
        return xc.load_config(p)

    def test_shipped_yaml_valid_and_feeds_meter_config(self):
        cfg = xc.load_config(HERE / "experiment_config_v52.yaml")
        mc = ep.MeterConfig.from_dict(xc.meter_dict(cfg))             # les noms de paramètres correspondent exactement
        self.assertEqual(mc.measure_power_secs, 1.0)
        self.assertEqual(cfg["paths"]["results_group"], "exp3_v52")
        self.assertIn("_sha256", cfg)
        lp = xc.loop_params(cfg)
        self.assertGreaterEqual(lp["min_seconds"], mc.min_window_s)

    def test_group_must_be_v52(self):
        for g in ("exp3_v51", "exp3_v5", "other"):
            with self.assertRaises(xc.ConfigError):
                self.load(lambda d, g=g: d["paths"].update(results_group=g))

    def test_dram_cannot_be_enabled(self):
        with self.assertRaises(xc.ConfigError):
            self.load(lambda d: d["energy"].update(rapl_include_dram=True))

    def test_unknown_keys_rejected(self):
        with self.assertRaises(xc.ConfigError):
            self.load(lambda d: d["energy"].update(typo_key=1))
        with self.assertRaises(xc.ConfigError):
            self.load(lambda d: d["energy"]["sampling"].update(measure_power_sec=1.0))

    def test_window_rules(self):
        with self.assertRaises(xc.ConfigError):                      # boucle plus courte que 10 périodes
            self.load(lambda d: d["energy"]["loop"].update(min_seconds=5.0))
        with self.assertRaises(xc.ConfigError):                      # idle trop court
            self.load(lambda d: d["energy"]["idle"].update(before_seconds=3.0))
        self.load(lambda d: d["energy"]["idle"].update(before_seconds=0.0, after_seconds=0.0))   # 0 = désactivé : accepté

    def test_value_rules(self):
        for mut in (lambda d: d["energy"]["sampling"].update(measure_power_secs=0),
                    lambda d: d["energy"]["sampling"].update(measure_power_secs="1"),
                    lambda d: d["energy"]["checks"]["load_idle_ratio"].update(severity="boom"),
                    lambda d: d["energy"].update(tracking_mode="gpu"),
                    lambda d: d.update(schema_version=51)):
            with self.assertRaises(xc.ConfigError):
                self.load(mut)

    def test_missing_file(self):
        with self.assertRaises(xc.ConfigError):
            xc.load_config(self.td / "absent.yaml")

    def test_planned_output_dir_is_under_v52_and_creates_nothing(self):
        cfg = xc.load_config(HERE / "experiment_config_v52.yaml")
        p = xc.planned_output_dir(cfg, self.td, "run1")
        self.assertEqual(p.parent.name, "exp3_v52")
        self.assertFalse(p.exists())
        with self.assertRaises(xc.ConfigError):
            xc.planned_output_dir(cfg, self.td, "../exp3_v51/x")


# ======================================================================================================
class Simulated_Meter(unittest.TestCase):
    """Backend SIMULÉ : ces tests valident la plomberie (horodatages, fenêtres, brut/net, boucle, journal), PAS l'énergie réelle."""

    def setUp(self):
        self.td = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def meter(self, backend=None, raw=True, cfg=FAST):
        return ep.EnergyMeter(backend or ep.FakeBackend(lambda n: (10.0, 1.0) if "idle" not in n else (3.0, 1.0)), cfg,
                              self.td / "raw.jsonl" if raw else None)

    def test_long_block_bracketed_by_idle(self):
        m = self.meter()
        value, b = m.measure_long("work", lambda: time.sleep(0.3) or 42, idle_before_s=0.25, idle_after_s=0.25)
        m.close()
        self.assertEqual(value, 42)
        self.assertEqual(b["baseline_source"], "bracketing_before_after")
        self.assertEqual(b["baseline"]["n_records"], 2)
        e, win = b["energy"], b["task"]["window_s"]
        self.assertAlmostEqual(e["net_cpu_j"], (10.0 - 3.0) * win, places=6)
        self.assertAlmostEqual(e["net_total_hybrid_j"], (11.0 - 4.0) * win, places=6)
        self.assertTrue(e["net_consistent"])
        self.assertTrue(b["valid"], [c for c in b["checks"] if c["status"] == "fail"])
        self.assertAlmostEqual(b["task"]["mean_power_cpu_w"], 10.0, places=6)
        self.assertTrue(math.isfinite(e["u_net_cpu_j"]))
        ep.dumps_strict(b)                                              # le bloc entier est sérialisable en JSON strict

    def test_calls_are_sequential_never_overlapping(self):
        be = ep.FakeBackend()
        m = self.meter(be, raw=False)
        m.measure_long("w", lambda: time.sleep(0.25), idle_before_s=0.25, idle_after_s=0.25)
        self.assertEqual(be.calls, ["start:w::idle_before", "stop:w::idle_before", "start:w", "stop:w",
                                    "start:w::idle_after", "stop:w::idle_after"])

    def test_nested_tasks_refused(self):
        m = self.meter(raw=False)
        with self.assertRaises(ep.EnergyError):
            with m.task("a"):
                with m.task("b"):
                    pass
        self.assertIsNone(m._active)

    def test_loop_reports_total_items_mean_and_time(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            time.sleep(0.02)
            return 1
        be = ep.FakeBackend(lambda n: (10.0, 1.0) if "idle" not in n else (3.0, 1.0))
        m = self.meter(be, raw=False)
        _, b = m.measure_loop("yolo", fn, min_seconds=0.3, warmup_iterations=3, idle_before_s=0.25, idle_after_s=0.25)
        loop, t, e = b["loop"], b["task"], b["energy"]
        self.assertGreaterEqual(loop["n_iterations"], 5)                 # marge pour la résolution du minuteur Windows (~15 ms)
        self.assertEqual(loop["n_items"], loop["n_iterations"])
        self.assertEqual(calls["n"], 3 + loop["n_iterations"])           # warmup exécuté HORS tâche mesurée
        self.assertEqual(loop["warmup_iterations"], 3)
        self.assertNotIn("start:yolo-warmup", be.calls)
        self.assertAlmostEqual(e["energy_total_hybrid_j_per_item"], t["energy_total_hybrid_j"] / loop["n_items"])
        self.assertAlmostEqual(loop["time_per_item_s"], t["wall_s"] / loop["n_items"])
        self.assertIn("MOYENNE", b["statement"])
        self.assertIn("pas la mesure d'un élément individuel", b["statement"])
        self.assertGreaterEqual(t["window_s"], 0.3 * 0.9)

    def test_loop_counts_items_returned_by_fn(self):
        m = self.meter(raw=False)
        _, b = m.measure_loop("batch", lambda: time.sleep(0.05) or 4, min_seconds=0.25, idle_before_s=0.25, idle_after_s=0.25)
        self.assertEqual(b["loop"]["n_items"], 4 * b["loop"]["n_iterations"])

    def test_loop_too_short_refused_unless_allowed(self):
        m = self.meter(raw=False)
        with self.assertRaises(ep.WindowTooShortError):
            m.measure_loop("short", lambda: 1, min_seconds=0.05)
        _, b = m.measure_loop("short", lambda: time.sleep(0.005) or 1, min_seconds=0.05, allow_short=True,
                              baseline=ep.idle_baseline([idle_rec(3.0), idle_rec(3.0)]))
        self.assertTrue(b["task"]["window_too_short"])
        self.assertFalse(b["valid"])                                       # flagrante : fenêtre < 4 périodes -> échec
        with self.assertRaises(ep.WindowTooShortError):
            m.measure_idle(0.05)

    def test_duration_mismatch_is_flagged(self):
        m = self.meter(ep.FakeBackend(duration_offset_s=0.1), raw=False)
        _, b = m.measure_long("w", lambda: time.sleep(0.3), baseline=ep.idle_baseline([idle_rec(3.0), idle_rec(3.0)]))
        self.assertGreater(b["task"]["window_gap_pct"], 2.0)
        self.assertIn("warn", [c["status"] for c in b["checks"] if c["check"] == "coherence_durees"])
        self.assertEqual(b["task"]["window_source"], "backend_duration_s")

    def test_idle_drift_and_cv_flagged(self):
        be = ep.FakeBackend(lambda n: (4.0, 1.0) if n.endswith("idle_after") else (3.0, 1.0) if "idle" in n else (10.0, 1.0))
        m = self.meter(be, raw=False)
        _, b = m.measure_long("w", lambda: time.sleep(0.25), idle_before_s=0.25, idle_after_s=0.25)
        self.assertAlmostEqual(b["idle_drift_relative"]["cpu"], 1 / 3, places=5)
        st = {c["check"]: c["status"] for c in b["checks"]}
        self.assertEqual(st["idle_derive_cpu"], "warn")
        self.assertEqual(st["idle_cv_cpu"], "warn")

    def test_negative_net_is_kept_in_block(self):
        be = ep.FakeBackend(lambda n: (2.0, 1.0) if "idle" not in n else (6.0, 1.0))
        m = self.meter(be, raw=False)
        _, b = m.measure_long("w", lambda: time.sleep(0.25), idle_before_s=0.25, idle_after_s=0.25)
        self.assertLess(b["energy"]["net_cpu_j"], 0)
        self.assertTrue(b["energy"]["net_cpu_negative"])
        self.assertAlmostEqual(b["energy"]["net_cpu_j"], (2.0 - 6.0) * b["task"]["window_s"], places=6)
        self.assertEqual(b["energy"]["net_cpu_negative_within_uncertainty"], False)   # u = bord seul, très inférieur à |net|

    def test_exception_in_body_recorded_and_meter_reusable(self):
        m = self.meter(raw=False)
        with self.assertRaises(RuntimeError):
            with m.task("boom"):
                raise RuntimeError("x")
        self.assertIn("RuntimeError", m.tasks[-1]["error"])
        self.assertFalse(m.tasks[-1]["valid"])
        self.assertIsNone(m._active)
        with m.task("after"):
            time.sleep(0.01)
        self.assertEqual(m.tasks[-1]["task"], "after")

    def test_raw_log_has_timestamps_and_strict_json(self):
        m = self.meter()
        m.measure_long("w", lambda: time.sleep(0.25), idle_before_s=0.25, idle_after_s=0.25)
        m.close()
        lines = [ep.loads_strict(x) for x in (self.td / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
        events = [l["event"] for l in lines]
        self.assertEqual(events[0], "meter_open")
        self.assertEqual(events.count("task"), 3)
        self.assertIn("block", events)
        self.assertEqual(events[-1], "meter_close")
        t = next(l for l in lines if l["event"] == "task" and l["task"] == "w")
        for k in ("t_start_mono_ns", "t_stop_mono_ns", "t_start_wall_ns", "t_stop_wall_ns", "wall_s", "backend_duration_s",
                  "window_s", "window_gap_s", "window_gap_pct", "backend_raw"):
            self.assertIn(k, t)
        self.assertGreater(t["t_stop_mono_ns"], t["t_start_mono_ns"])
        self.assertGreaterEqual(t["t_stop_wall_ns"], t["t_start_wall_ns"])
        self.assertAlmostEqual(t["wall_s"], (t["t_stop_mono_ns"] - t["t_start_mono_ns"]) / 1e9, places=9)
        self.assertEqual(t["n_items"], 1)

    def test_loop_raw_log_n_items_consistent(self):
        m = self.meter()
        _, b = m.measure_loop("l", lambda: time.sleep(0.02) or 1, min_seconds=0.25, idle_before_s=0.25, idle_after_s=0.25)
        m.close()
        lines = [ep.loads_strict(x) for x in (self.td / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
        t = next(l for l in lines if l["event"] == "task" and l["task"] == "l")
        self.assertEqual(t["n_items"], b["loop"]["n_items"])
        self.assertTrue(any(l["event"] == "loop_summary" for l in lines))

    def test_raw_log_not_overwritten(self):
        self.meter().close()
        with self.assertRaises(FileExistsError):
            self.meter()

    def test_simulated_source_is_never_reported_as_measured(self):
        m = self.meter(raw=False)
        m.measure_idle(0.25)
        info = m.method_info()
        self.assertTrue(info["simulated"])
        self.assertFalse(info["cpu_is_confirmed_rapl_measurement"])
        self.assertEqual(info["cpu_energy_source"], "simulated_not_hardware")
        self.assertEqual(info["ram_energy_source"], "modelled")
        self.assertEqual(info["primary_energy_metric"], "none_confirmed")
        self.assertFalse(info["raw_samples_exposed"])
        s = m.summary()
        self.assertEqual(s["n_tasks"], 1)
        ep.dumps_strict(s)

    def test_cpu_zero_flagged(self):
        m = self.meter(ep.FakeBackend(cpu_zero=True), raw=False)
        m.measure_idle(0.25)
        self.assertEqual(m.cpu_nonpositive_tasks, ["idle"])
        self.assertIn("warn", [c["status"] for c in m.tasks[-1]["checks"] if c["check"] == "energie_cpu_positive"])

    def test_energy_columns_separate_cpu_ram_total(self):
        m = self.meter(raw=False)
        _, b = m.measure_long("w", lambda: time.sleep(0.25), baseline=ep.idle_baseline([idle_rec(3.0), idle_rec(3.0)]))
        t = b["task"]
        for k in ("energy_cpu_j", "energy_ram_modelled_j", "energy_total_hybrid_j"):
            self.assertIn(k, t)
        self.assertNotIn("energy_ram_j", t)                                 # la RAM n'apparaît jamais sous un nom neutre
        self.assertAlmostEqual(t["energy_total_hybrid_j"], t["energy_cpu_j"] + t["energy_ram_modelled_j"] + t["energy_gpu_j"])


# ======================================================================================================
def _hw_enabled():
    if sys.platform != "win32" or os.environ.get("V52_HARDWARE") != "1":
        return False
    import importlib.util
    return importlib.util.find_spec("codecarbon") is not None


@unittest.skipUnless(_hw_enabled(), "NON EXÉCUTÉ : test matériel Windows (V52_HARDWARE=1 + Windows + codecarbon requis)")
class Hardware_WindowsRapl(unittest.TestCase):
    """À lancer sur le PC Windows, machine au repos, secteur branché, aucune autre expérience en cours."""

    def test_codecarbon_backend_idle_and_busy(self):
        cfg = xc.load_config(HERE / "experiment_config_v52.yaml")
        mc = ep.MeterConfig.from_dict(xc.meter_dict(cfg))
        td = Path(tempfile.mkdtemp())
        m = ep.new_codecarbon_meter(mc, td / "raw.jsonl")
        try:
            def busy():
                t0, x = time.perf_counter(), 0.0
                while time.perf_counter() - t0 < mc.min_window_s * 2:
                    x = (x + math.sqrt(x + 1.0)) % 1e6
            _, b = m.measure_long("busy", busy, idle_before_s=mc.min_window_s, idle_after_s=mc.min_window_s)
            info = m.method_info()
        finally:
            m.close()
        print("\n[HW] source CPU :", info.get("cpu_energy_source"), "| canaux :", info.get("rapl_channels_selected"))
        print("[HW] P_idle CPU : %.3f W (CV %.4f) | P_charge CPU : %.3f W | net CPU : %.3f J +- %.3f J" % (
            b["baseline"]["cpu"]["mean_w"], b["baseline"]["cpu"]["cv"], b["task"]["mean_power_cpu_w"],
            b["energy"]["net_cpu_j"], b["energy"]["u_net_cpu_j"]))
        print("[HW] contrôles non ok :", [c for c in b["checks"] if c["status"] != "ok"])
        self.assertTrue(info["cpu_is_confirmed_rapl_measurement"], f"RAPL non confirmé : {info.get('cpu_energy_source')}")
        self.assertEqual(info["ram_energy_source"], "modelled")
        shutil.rmtree(td, ignore_errors=True)

    def test_probe_api_report_shape(self):
        r = ep.probe_codecarbon_api()
        self.assertTrue(r.get("available"), r)
        print("\n[HW] sonde API CodeCarbon :\n" + ep.dumps_strict(r))


# ======================================================================================================
class _CategoryResult(unittest.TextTestResult):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.cat = {}

    def _c(self, test):
        n = type(test).__name__
        return n.split("_")[0] if "_" in n else "Autre"

    def _bump(self, test, key):
        self.cat.setdefault(self._c(test), {"réussi": 0, "échec": 0, "non exécuté": 0})[key] += 1

    def addSuccess(self, test):
        super().addSuccess(test)
        self._bump(test, "réussi")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._bump(test, "échec")

    def addError(self, test, err):
        super().addError(test, err)
        self._bump(test, "échec")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._bump(test, "non exécuté")


def main():
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    res = unittest.TextTestRunner(verbosity=2, resultclass=_CategoryResult).run(suite)
    labels = {"Unit": "Tests unitaires (sans backend)", "Simulated": "Tests avec données SIMULÉES (FakeBackend)",
              "Hardware": "Tests matériels Windows"}
    print("\n" + "=" * 78 + "\nBILAN PAR CATÉGORIE\n" + "=" * 78)
    for k in ("Unit", "Simulated", "Hardware"):
        c = res.cat.get(k, {"réussi": 0, "échec": 0, "non exécuté": 0})
        print(f"{labels[k]:<46} réussis={c['réussi']:<3} échecs={c['échec']:<3} NON EXÉCUTÉS={c['non exécuté']}")
    print("Aucun résultat matériel n'est simulé : la validation énergétique RAPL n'est acquise que si la catégorie matérielle "
          "est exécutée ET réussie.")
    sys.exit(0 if res.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
