# -*- coding: utf-8 -*-
"""
exp_config_v52.py -- lecteur et validateur de experiment_config_v52.yaml (V5.2). AUTONOME : n'importe ni exp_config_v51 ni
energy_profiling_v5x ; bibliothèque standard + PyYAML (import paresseux). Ne lance aucune mesure, ne crée aucun dossier.
"""
from __future__ import annotations

import copy
import hashlib
import math
import re
from pathlib import Path
from typing import Any, Dict

SCHEMA_VERSION = 52
REQUIRED_RESULTS_GROUP = "exp3_v52"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


class ConfigError(Exception):
    pass


DEFAULTS: Dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "paths": {"results_root": "results/VCM_EnergyMap", "results_group": REQUIRED_RESULTS_GROUP},
    "energy": {
        "country_iso_code": "MAR", "tracking_mode": "machine", "expected_channel": "RAPL_Package0_PKG",
        "require_confirmed_rapl": True, "rapl_include_dram": False,
        "sampling": {"measure_power_secs": 1.0, "min_window_periods": 10.0},
        "idle": {"before_seconds": 30.0, "after_seconds": 30.0, "cooldown_seconds": 30.0, "min_records": 2,
                 "cv_max": 0.10, "drift_max_rel": 0.10},
        "checks": {"duration_tolerance_pct": 2.0,
                   "load_idle_ratio": {"enabled": True, "min_ratio": 1.2, "severity": "warn"}},
        "uncertainty": {"significance_k": 2.0, "boundary": {"enabled": True, "half_width_periods": 1.0}},
        "loop": {"min_seconds": 20.0, "min_iterations": 1, "warmup_iterations": 2},
        "raw_log": {"enabled": True},
    },
}


def _fill(defaults: Any, user: Any, path: str = "") -> Any:
    """Fusion récursive ; une clé absente des valeurs par défaut est une ERREUR (faute de frappe)."""
    if not isinstance(defaults, dict):
        return copy.deepcopy(user)
    if user is None:
        return copy.deepcopy(defaults)
    if not isinstance(user, dict):
        raise ConfigError(f"'{path or 'racine'}' doit être un dictionnaire")
    unknown = set(user) - set(defaults)
    if unknown:
        raise ConfigError(f"clé(s) inconnue(s) sous '{path or 'racine'}' : {sorted(unknown)}")
    return {k: _fill(defaults[k], user[k], f"{path}.{k}" if path else k) if k in user else copy.deepcopy(defaults[k])
            for k in defaults}


def _num(v: Any, name: str, *, gt: float = None, ge: float = None) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ConfigError(f"{name} doit être un nombre fini (reçu {v!r})")
    if gt is not None and not v > gt:
        raise ConfigError(f"{name} doit être > {gt} (reçu {v})")
    if ge is not None and not v >= ge:
        raise ConfigError(f"{name} doit être >= {ge} (reçu {v})")
    return float(v)


def _bool(v: Any, name: str) -> bool:
    if not isinstance(v, bool):
        raise ConfigError(f"{name} doit être un booléen (reçu {v!r})")
    return v


def validate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Valide les valeurs (types, bornes, cohérences). Retourne cfg ; lève ConfigError."""
    if cfg.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(f"schema_version attendu {SCHEMA_VERSION}, reçu {cfg.get('schema_version')!r}")
    P, E = cfg["paths"], cfg["energy"]
    if P["results_group"] != REQUIRED_RESULTS_GROUP:
        raise ConfigError(f"paths.results_group doit valoir '{REQUIRED_RESULTS_GROUP}' (reçu {P['results_group']!r}) : "
                          f"isolation des résultats V5.2")
    if not isinstance(P["results_root"], str) or not P["results_root"].strip():
        raise ConfigError("paths.results_root doit être une chaîne non vide")
    if _bool(E["rapl_include_dram"], "energy.rapl_include_dram"):
        raise ConfigError("energy.rapl_include_dram doit rester false (double comptage avec la RAM modélisée)")
    _bool(E["require_confirmed_rapl"], "energy.require_confirmed_rapl")
    if E["tracking_mode"] not in ("machine", "process"):
        raise ConfigError(f"energy.tracking_mode inconnu : {E['tracking_mode']!r} (machine | process)")
    if not isinstance(E["country_iso_code"], str) or len(E["country_iso_code"]) != 3:
        raise ConfigError("energy.country_iso_code : code ISO alpha-3 attendu")
    S, I, C, U, L = E["sampling"], E["idle"], E["checks"], E["uncertainty"], E["loop"]
    period = _num(S["measure_power_secs"], "energy.sampling.measure_power_secs", gt=0)
    periods = _num(S["min_window_periods"], "energy.sampling.min_window_periods", gt=0)
    min_win = period * periods
    for k in ("before_seconds", "after_seconds"):
        v = _num(I[k], f"energy.idle.{k}", ge=0)
        if 0 < v < min_win:
            raise ConfigError(f"energy.idle.{k}={v:g} s < fenêtre minimale {min_win:g} s ({periods:g} x {period:g} s)")
    _num(I["cooldown_seconds"], "energy.idle.cooldown_seconds", ge=0)
    if not isinstance(I["min_records"], int) or isinstance(I["min_records"], bool) or I["min_records"] < 1:
        raise ConfigError("energy.idle.min_records doit être un entier >= 1")
    _num(I["cv_max"], "energy.idle.cv_max", gt=0)
    _num(I["drift_max_rel"], "energy.idle.drift_max_rel", gt=0)
    _num(C["duration_tolerance_pct"], "energy.checks.duration_tolerance_pct", gt=0)
    R = C["load_idle_ratio"]
    _bool(R["enabled"], "energy.checks.load_idle_ratio.enabled")
    _num(R["min_ratio"], "energy.checks.load_idle_ratio.min_ratio", gt=0)
    if R["severity"] not in ("info", "warn", "fail"):
        raise ConfigError("energy.checks.load_idle_ratio.severity : info | warn | fail")
    _num(U["significance_k"], "energy.uncertainty.significance_k", gt=0)
    _bool(U["boundary"]["enabled"], "energy.uncertainty.boundary.enabled")
    _num(U["boundary"]["half_width_periods"], "energy.uncertainty.boundary.half_width_periods", ge=0)
    ms = _num(L["min_seconds"], "energy.loop.min_seconds", gt=0)
    if ms < min_win:
        raise ConfigError(f"energy.loop.min_seconds={ms:g} s < fenêtre minimale {min_win:g} s : la boucle doit couvrir "
                          f"{periods:g} périodes d'échantillonnage au moins")
    for k in ("min_iterations", "warmup_iterations"):
        if not isinstance(L[k], int) or isinstance(L[k], bool) or L[k] < (1 if k == "min_iterations" else 0):
            raise ConfigError(f"energy.loop.{k} : entier invalide ({L[k]!r})")
    _bool(E["raw_log"]["enabled"], "energy.raw_log.enabled")
    return cfg


def load_config(path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ConfigError("PyYAML absent : python -m pip install pyyaml") from e
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"fichier de configuration introuvable : {p}")
    raw = p.read_bytes()
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ConfigError("le YAML doit contenir un dictionnaire à la racine")
    cfg = validate(_fill(DEFAULTS, data))
    cfg["_file"], cfg["_sha256"] = str(p.resolve()), hashlib.sha256(raw).hexdigest()
    return cfg


def meter_dict(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Paramètres à plat pour energy_profiling_v52.MeterConfig.from_dict (les noms correspondent exactement)."""
    E = cfg["energy"]
    return {"country_iso_code": E["country_iso_code"], "tracking_mode": E["tracking_mode"],
            "measure_power_secs": float(E["sampling"]["measure_power_secs"]),
            "min_window_periods": float(E["sampling"]["min_window_periods"]),
            "duration_tolerance_pct": float(E["checks"]["duration_tolerance_pct"]),
            "load_idle_ratio_check": E["checks"]["load_idle_ratio"]["enabled"],
            "load_idle_ratio_min": float(E["checks"]["load_idle_ratio"]["min_ratio"]),
            "load_idle_ratio_severity": E["checks"]["load_idle_ratio"]["severity"],
            "idle_cv_max": float(E["idle"]["cv_max"]), "idle_min_records": int(E["idle"]["min_records"]),
            "idle_drift_max_rel": float(E["idle"]["drift_max_rel"]),
            "significance_k": float(E["uncertainty"]["significance_k"]),
            "boundary_enabled": E["uncertainty"]["boundary"]["enabled"],
            "boundary_half_width_periods": float(E["uncertainty"]["boundary"]["half_width_periods"]),
            "expected_channel": E["expected_channel"], "require_confirmed_rapl": E["require_confirmed_rapl"]}


def idle_plan(cfg: Dict[str, Any]) -> Dict[str, float]:
    I = cfg["energy"]["idle"]
    return {"before_seconds": float(I["before_seconds"]), "after_seconds": float(I["after_seconds"]),
            "cooldown_seconds": float(I["cooldown_seconds"])}


def loop_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    L = cfg["energy"]["loop"]
    return {"min_seconds": float(L["min_seconds"]), "min_iterations": int(L["min_iterations"]),
            "warmup_iterations": int(L["warmup_iterations"])}


def planned_output_dir(cfg: Dict[str, Any], project_root, dirname: str) -> Path:
    """Chemin PRÉVU d'un dossier de résultats (ne crée rien). Refuse un nom non sûr. Le refus d'écrasement et la création
    sont assurés par energy_profiling_v52.resolve_output_dir."""
    if not isinstance(dirname, str) or not _SAFE_NAME.match(dirname) or dirname in (".", ".."):
        raise ConfigError(f"nom de dossier non sûr : {dirname!r}")
    root = Path(cfg["paths"]["results_root"])
    root = root if root.is_absolute() else Path(project_root) / root
    return root / cfg["paths"]["results_group"] / dirname
