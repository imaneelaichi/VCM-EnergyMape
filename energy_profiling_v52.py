# -*- coding: utf-8 -*-
"""
energy_profiling_v52.py -- module énergétique AUTONOME de VCM-EnergyMap V5.2.

Indépendant de V5.1 : n'importe NI energy_profiling_v51 NI exp_config_v51 (ni aucun module du codec VCM, des checkpoints
ou de YOLO). Bibliothèque standard seule ; `codecarbon` (3.3.2 visé) et `psutil` ne sont importés que paresseusement.

CE QUI EST (ET N'EST PAS) MESURÉ
--------------------------------
* CPU   : énergie du PACKAGE CPU. Considérée "mesurée RAPL" UNIQUEMENT si la source est confirmée (backend windows_emi,
          un seul canal Package, compteur ayant délivré > 0 J sur toutes les tâches). Sinon elle est étiquetée
          explicitement (modélisée / non confirmée / simulée).
* RAM   : TOUJOURS modélisée (puissance estimée par CodeCarbon x durée). Jamais une mesure matérielle. Colonne dédiée
          `energy_ram_modelled_j`.
* Total : `energy_total_hybrid_j` = CPU (mesuré si confirmé) + RAM (modélisée) [+ GPU]. Hybride, jamais "mesuré".
* Boucle: pour une phase courte, on mesure UNE boucle de N appels. On rapporte l'énergie totale, N, l'énergie MOYENNE
          par élément et le temps moyen par élément. Ce n'est PAS l'énergie d'une inférence individuelle.

DONNÉES BRUTES (limitation explicite)
-------------------------------------
Le module journalise, pour chaque tâche, les valeurs rendues par `stop_task()` du backend et quatre horodatages
(monotonique et mural, début et fin) dans un JSONL (`RawLog`). Il N'AFFIRME PAS que les échantillons bruts RAPL/EMI sont
exposés par CodeCarbon : `CodeCarbonBackend.describe()` indique `raw_samples_exposed = False` et
`probe_codecarbon_api()` (à lancer sur Windows) liste ce que l'API expose réellement.

INCERTITUDE (modèle documenté, hypothèses explicites)
-----------------------------------------------------
net_c = E_c - P_idle_c x fenêtre.
u_idle  = erreur-type de la moyenne des puissances idle x fenêtre (défini seulement si >= 2 mesures idle).
u_bord  = (demi-largeur = h x période d'échantillonnage x puissance moyenne de la tâche) / sqrt(3) (loi rectangulaire ;
          h configurable) : incertitude de bord liée à l'échantillonnage. HYPOTHÈSE de modèle, pas une propriété prouvée.
u_net   = sqrt(u_idle^2 + u_bord^2) ; None si u_idle n'est pas défini.
Les nets NÉGATIFS sont conservés tels quels ; on signale s'ils sont compatibles avec zéro (|net| <= k x u).
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

MODULE_VERSION = "v52"
REQUIRED_RESULTS_GROUP = "exp3_v52"
KWH_TO_J = 3.6e6
NAN = float("nan")

# composant interne -> suffixe de nom de colonne
KEY = {"cpu": "cpu", "ram": "ram_modelled", "gpu": "gpu", "total": "total_hybrid"}
COMPS = ("cpu", "ram", "gpu", "total")


class EnergyError(Exception):
    pass


class IsolationError(EnergyError):
    pass


class WindowTooShortError(EnergyError):
    pass


# =====================================================================================
# 1. Sérialisation JSON stricte
# =====================================================================================
def isfin(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def json_safe(o: Any) -> Any:
    """Convertit récursivement en structure JSON STRICTE : NaN / +-Infinity -> None ; clés -> str ; Path -> str ;
    dataclass -> dict ; set -> liste triée ; scalaires/tableaux numpy -> types Python."""
    if o is None or isinstance(o, (str, bool)):
        return o
    if isinstance(o, int):
        return int(o)
    if isinstance(o, float):
        return float(o) if math.isfinite(o) else None
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return json_safe(dataclasses.asdict(o))
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (set, frozenset)):
        return [json_safe(v) for v in sorted(o, key=str)]
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "tolist") and callable(o.tolist):          # numpy
        try:
            return json_safe(o.tolist())
        except Exception:                                    # noqa: BLE001
            pass
    return str(o)


def dumps_strict(obj: Any, indent: Optional[int] = 2) -> str:
    """JSON conforme RFC 8259 : `allow_nan=False` (lève si une valeur non finie avait échappé à json_safe)."""
    return json.dumps(json_safe(obj), indent=indent, ensure_ascii=False, allow_nan=False)


def _reject_constant(c):
    raise ValueError(f"constante JSON non standard : {c}")


def loads_strict(text: str) -> Any:
    """Parse strict (refuse NaN / Infinity / -Infinity)."""
    return json.loads(text, parse_constant=_reject_constant)


def dump_json_strict(path, obj: Any, overwrite: bool = False) -> Path:
    """Écriture atomique ; REFUSE d'écraser un fichier existant sauf overwrite=True."""
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"refus d'écraser : {p}")
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(dumps_strict(obj), encoding="utf-8")
    os.replace(tmp, p)
    return p


def csv_cell(v: Any) -> Any:
    """Valeur de cellule CSV : non fini -> chaîne vide (jamais 'nan'/'inf')."""
    if isinstance(v, float) and not math.isfinite(v):
        return ""
    return v


class RawLog:
    """JSONL append-only. Ouverture en mode exclusif ('x') : refuse d'écraser un journal existant."""

    def __init__(self, path):
        self.path = Path(path)
        self._f = open(self.path, "x", encoding="utf-8")

    def append(self, record: Mapping[str, Any]) -> None:
        self._f.write(json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()


# =====================================================================================
# 2. Isolation des résultats
# =====================================================================================
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_OTHER_VERSION = re.compile(r"exp3_v(?!52\b)\w*", re.IGNORECASE)


def resolve_output_dir(results_root, results_group: str, dirname: str, *, create: bool = True,
                       required_group: str = REQUIRED_RESULTS_GROUP) -> Path:
    """Dossier de résultats d'UNE expérience : <results_root>/<results_group>/<dirname>.
    Refuse : groupe != exp3_v52 ; nom dangereux (séparateurs, '..') ; sortie de results_root ; toute référence à un autre
    groupe exp3_v* ; dossier DÉJÀ existant (même vide). Ne supprime et n'écrase jamais rien."""
    if results_group != required_group:
        raise IsolationError(f"results_group={results_group!r} != {required_group!r} : résultats V5.2 isolés uniquement")
    if not isinstance(dirname, str) or not _SAFE_NAME.match(dirname) or dirname in (".", ".."):
        raise IsolationError(f"nom de dossier non sûr : {dirname!r}")
    if _OTHER_VERSION.search(dirname):
        raise IsolationError(f"le nom {dirname!r} référence un autre groupe de résultats")
    root = Path(results_root).resolve()
    target = (root / results_group / dirname).resolve()
    try:
        target.relative_to(root / results_group)
    except ValueError as e:
        raise IsolationError(f"{target} sort de {root / results_group}") from e
    if target.exists():
        raise FileExistsError(f"le dossier existe déjà, rien n'est écrasé : {target}")
    if create:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(exist_ok=False)
    return target


# =====================================================================================
# 3. Fonctions pures : fenêtres, puissance, baseline, brut/net, incertitude
# =====================================================================================
def mean_power_w(energy_j: float, window_s: float) -> float:
    """Puissance moyenne = énergie / fenêtre. NaN si une entrée n'est pas finie ou si la fenêtre n'est pas > 0."""
    if isfin(energy_j) and isfin(window_s) and window_s > 0:
        return float(energy_j) / float(window_s)
    return NAN


def choose_window(cc_duration_s, wall_s) -> Tuple[float, str]:
    """Fenêtre d'énergie : durée du backend (même horloge que l'intégration) si valide, sinon durée réelle (repli signalé)."""
    if isfin(cc_duration_s) and cc_duration_s > 0:
        return float(cc_duration_s), "backend_duration_s"
    if isfin(wall_s) and wall_s > 0:
        return float(wall_s), "wall_s_fallback"
    return NAN, "invalid"


def window_gap(wall_s, window_s) -> Tuple[float, float]:
    """(écart en s = wall - fenêtre, écart en % de la fenêtre)."""
    if isfin(wall_s) and isfin(window_s) and window_s > 0:
        return float(wall_s) - float(window_s), 100.0 * abs(float(wall_s) - float(window_s)) / float(window_s)
    return NAN, NAN


def _stats(v: Sequence[float]) -> Dict[str, Any]:
    v = [float(x) for x in v if isfin(x)]
    if not v:
        return {"n": 0, "mean_w": NAN, "sd_w": NAN, "sem_w": NAN, "cv": NAN, "min_w": NAN, "max_w": NAN}
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else NAN
    return {"n": len(v), "mean_w": m, "sd_w": sd, "sem_w": sd / math.sqrt(len(v)) if isfin(sd) else NAN,
            "cv": sd / abs(m) if isfin(sd) and m != 0 else NAN, "min_w": min(v), "max_w": max(v)}


def idle_baseline(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Baseline à vide poolée : statistiques des puissances moyennes (W) par composant sur TOUTES les mesures idle fournies
    (typiquement avant + après le bloc). Aucune mesure n'est écartée."""
    out: Dict[str, Any] = {"n_records": len(records)}
    for c in COMPS:
        out[c] = _stats([r.get(f"mean_power_{KEY[c]}_w", NAN) for r in records])
    return out


def idle_drift(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, float]:
    """Dérive relative de la puissance idle entre deux mesures : (après - avant) / avant, par composant."""
    d = {}
    for c in ("cpu", "total"):
        b, a = before.get(f"mean_power_{KEY[c]}_w", NAN), after.get(f"mean_power_{KEY[c]}_w", NAN)
        d[c] = (a - b) / b if isfin(a) and isfin(b) and b != 0 else NAN
    return d


def compute_energy(rec: Mapping[str, Any], baseline: Mapping[str, Any], *, sampling_period_s: float,
                   boundary_enabled: bool = True, boundary_half_width_periods: float = 1.0,
                   significance_k: float = 2.0, n_items: int = 1) -> Dict[str, Any]:
    """Énergie BRUTE / NETTE par composant avec incertitude (voir docstring du module).

    rec      : energy_{cpu,ram_modelled,gpu,total_hybrid}_j et window_s
    baseline : sortie de idle_baseline()
    Les nets négatifs sont conservés ; jamais de clamp, jamais de valeur absolue."""
    win = rec.get("window_s", NAN)
    out: Dict[str, Any] = {"window_s": win}
    nets: Dict[str, float] = {}
    for c in COMPS:
        k = KEY[c]
        e = rec.get(f"energy_{k}_j", NAN)
        bs = baseline.get(c, {}) if baseline else {}
        p_idle, sem = bs.get("mean_w", NAN), bs.get("sem_w", NAN)
        net = e - p_idle * win if isfin(e) and isfin(p_idle) and isfin(win) else NAN
        u_idle = sem * win if isfin(sem) and isfin(win) else NAN
        p_task = mean_power_w(e, win)
        u_b = (boundary_half_width_periods * sampling_period_s * abs(p_task) / math.sqrt(3.0)
               if boundary_enabled and isfin(p_task) else (0.0 if not boundary_enabled else NAN))
        u_net = math.sqrt(u_idle ** 2 + u_b ** 2) if isfin(u_idle) and isfin(u_b) else NAN
        nets[c] = net
        out.update({f"energy_{k}_j": e, f"idle_power_{k}_w": p_idle,
                    f"idle_energy_{k}_j": p_idle * win if isfin(p_idle) and isfin(win) else NAN,
                    f"net_{k}_j": net, f"u_idle_{k}_j": u_idle, f"u_boundary_{k}_j": u_b, f"u_net_{k}_j": u_net,
                    f"net_{k}_negative": bool(isfin(net) and net < 0),
                    f"net_{k}_significant": (bool(abs(net) > significance_k * u_net) if isfin(net) and isfin(u_net) else None),
                    f"net_{k}_negative_within_uncertainty":
                        (bool(net < 0 and abs(net) <= significance_k * u_net) if isfin(net) and isfin(u_net) else None)})
    # contrôle : net total direct (E_total - P_idle_total x fenêtre) vs somme des nets des composants
    comp_sum = sum(nets[c] for c in ("cpu", "ram", "gpu") if isfin(nets[c]))
    direct = nets["total"]
    diff = comp_sum - direct if isfin(direct) else NAN
    scale = max(1.0, abs(rec.get("energy_total_hybrid_j", 0.0) or 0.0), abs(out.get("idle_energy_total_hybrid_j", 0.0) or 0.0))
    out["net_component_sum_j"] = comp_sum
    out["net_component_sum_difference_j"] = diff
    out["net_consistent"] = bool(isfin(diff) and abs(diff) <= 1e-9 + 1e-9 * scale)
    n = max(int(n_items), 1)
    for c in COMPS:
        k = KEY[c]
        out[f"energy_{k}_j_per_item"] = out[f"energy_{k}_j"] / n if isfin(out[f"energy_{k}_j"]) else NAN
        out[f"net_{k}_j_per_item"] = out[f"net_{k}_j"] / n if isfin(out[f"net_{k}_j"]) else NAN
        out[f"u_net_{k}_j_per_item"] = out[f"u_net_{k}_j"] / n if isfin(out[f"u_net_{k}_j"]) else NAN
    out["uncertainty_model"] = ("u_net = sqrt(u_idle^2 + u_boundary^2) ; u_idle = erreur-type de la moyenne idle x fenêtre ; "
                                f"u_boundary = h x période x P_tâche / sqrt(3) (h={boundary_half_width_periods}, "
                                f"{'activé' if boundary_enabled else 'désactivé'}) ; hypothèse de modèle")
    return out


# =====================================================================================
# 4. Classification des sources (mesurée / modélisée / hybride)
# =====================================================================================
def is_package_channel_name(name: str) -> bool:
    n = str(name).lower()
    return "pkg" in n and not any(t in n for t in ("dram", "pp0", "pp1", "core", "psys"))


def classify_sources(cpu_mode: Optional[str], emi_probe: Optional[Mapping[str, Any]], n_tasks: int,
                     n_cpu_nonpositive: int, expected_channel: str = "", simulated: bool = False) -> Dict[str, Any]:
    """Classification pure (testable sans CodeCarbon). La source CPU n'est 'confirmée RAPL' que si : backend windows_emi,
    UN SEUL canal sélectionné et c'est un canal Package (éventuellement le canal attendu), et le compteur a délivré > 0 J sur
    TOUTES les tâches mesurées. La RAM est toujours modélisée."""
    mode = str(cpu_mode).lower()
    probe = dict(emi_probe or {})
    names = list(probe.get("selected_channel_names") or [])
    pkg_only = bool(names) and all(is_package_channel_name(n) for n in names)
    single = len(names) == 1
    expected_ok = (not expected_channel) or (expected_channel.lower() in [n.lower() for n in names])
    channel_ok = bool(probe.get("probe_ok")) and pkg_only and single and expected_ok
    functional = None if (n_tasks == 0 or mode != "windows_emi") else (n_cpu_nonpositive == 0)
    if simulated:
        src = "simulated_not_hardware"
    elif mode == "windows_emi":
        if not channel_ok:
            src = "windows_emi_package_channel_NOT_confirmed"
        elif functional is None:
            src = "windows_emi_package_channel_confirmed_energy_not_yet_checked"
        elif functional:
            src = "measured_rapl_via_windows_emi"
        else:
            src = "windows_emi_package_channel_confirmed_but_counter_returned_no_energy"
    elif mode == "intel_rapl":
        src = "intel_rapl_powercap_domain_not_verified_by_this_module"
    elif mode == "cpu_load":
        src = "modelled_tdp_x_load"
    elif mode == "constant":
        src = "modelled_constant_tdp"
    else:
        src = f"unclassified_mode_{mode}"
    confirmed = src == "measured_rapl_via_windows_emi"
    return {"cpu_backend_mode": cpu_mode, "cpu_energy_source": src, "cpu_is_confirmed_rapl_measurement": confirmed,
            "cpu_counter_functional": functional, "rapl_channels_selected": names,
            "rapl_package_channel_confirmed": channel_ok, "rapl_include_dram": probe.get("include_dram"),
            "ram_energy_source": "modelled",
            "ram_energy_detail": "puissance RAM estimée par CodeCarbon x durée ; jamais une mesure RAPL/DRAM",
            "total_energy_kind": ("hybrid_cpu_measured_rapl_plus_ram_modelled" if confirmed
                                  else ("simulated_not_a_measurement" if simulated else "not_a_confirmed_hardware_measurement")),
            "primary_energy_metric": "cpu_package_rapl" if confirmed else "none_confirmed"}


# =====================================================================================
# 5. Configuration du gestionnaire
# =====================================================================================
@dataclass(frozen=True)
class MeterConfig:
    country_iso_code: str = "MAR"
    tracking_mode: str = "machine"
    measure_power_secs: float = 1.0
    min_window_periods: float = 10.0              # fenêtre minimale = min_window_periods x measure_power_secs
    duration_tolerance_pct: float = 2.0           # écart toléré fenêtre backend / durée réelle
    load_idle_ratio_check: bool = True            # critère de CONTRÔLE configurable, pas une vérité universelle
    load_idle_ratio_min: float = 1.2
    load_idle_ratio_severity: str = "warn"        # info | warn | fail
    idle_cv_max: float = 0.10
    idle_min_records: int = 2
    idle_drift_max_rel: float = 0.10
    significance_k: float = 2.0
    boundary_enabled: bool = True
    boundary_half_width_periods: float = 1.0
    expected_channel: str = "RAPL_Package0_PKG"
    require_confirmed_rapl: bool = True

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MeterConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise EnergyError(f"paramètres inconnus pour MeterConfig : {sorted(unknown)}")
        return cls(**dict(d))

    @property
    def min_window_s(self) -> float:
        return self.min_window_periods * self.measure_power_secs


# =====================================================================================
# 6. Backends
# =====================================================================================
class Backend:
    """Contrat d'un backend énergétique."""
    name = "abstract"
    simulated = False

    def start_task(self, name: str) -> None:
        raise NotImplementedError

    def stop_task(self) -> Dict[str, Any]:
        """Retourne : energy_total_j, energy_cpu_j, energy_ram_j, energy_gpu_j, duration_s, cpu_power_w_last, ram_power_w_last"""
        raise NotImplementedError

    def describe(self, n_tasks: int, n_cpu_nonpositive: int, expected_channel: str) -> Dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class CodeCarbonBackend(Backend):
    """Adaptateur CodeCarbon 3.3.2 (OfflineEmissionsTracker). Les noms d'attributs de `stop_task()` et la sonde EMI sont ceux
    utilisés par V5.1 ; ils N'ONT PAS été revérifiés ici (CodeCarbon absent de l'environnement de développement).
    L'introspection de la sélection de canaux EMI repose sur des attributs PRIVÉS de CodeCarbon : API non garantie stable."""
    name = "codecarbon"
    _ACTIVE: Optional["CodeCarbonBackend"] = None

    def __init__(self, cfg: MeterConfig):
        if CodeCarbonBackend._ACTIVE is not None:
            raise EnergyError("un CodeCarbonBackend est déjà actif : un seul tracker à la fois")
        from codecarbon import OfflineEmissionsTracker      # import paresseux
        self.cfg = cfg
        self.t = OfflineEmissionsTracker(country_iso_code=cfg.country_iso_code, project_name="VCM-EnergyMap-v52",
                                         measure_power_secs=cfg.measure_power_secs, save_to_file=False, log_level="error",
                                         tracking_mode=cfg.tracking_mode, rapl_include_dram=False)
        self.t.start()
        CodeCarbonBackend._ACTIVE = self

    def start_task(self, name):
        self.t.start_task(name)

    def stop_task(self):
        d = self.t.stop_task()
        if d is None:
            raise EnergyError("CodeCarbon stop_task() a retourné None")

        def g(attr):
            v = getattr(d, attr, None)
            return float(v) if isfin(v) else NAN
        return {"energy_total_j": g("energy_consumed") * KWH_TO_J, "energy_cpu_j": g("cpu_energy") * KWH_TO_J,
                "energy_ram_j": g("ram_energy") * KWH_TO_J,
                "energy_gpu_j": (g("gpu_energy") * KWH_TO_J) if isfin(g("gpu_energy")) else 0.0,
                "duration_s": g("duration"), "cpu_power_w_last": g("cpu_power"), "ram_power_w_last": g("ram_power")}

    def _cpu_hw(self):
        for h in getattr(self.t, "_hardware", []):
            if type(h).__name__ == "CPU":
                return h
        return None

    def probe_emi(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"probe_ok": False}
        h = self._cpu_hw()
        iface = getattr(h, "_intel_interface", None) if h is not None else None
        if iface is None:
            out["probe_error"] = "CPU._intel_interface absent (mode non windows_emi/intel_rapl, ou API modifiée)"
            return out
        devices = getattr(iface, "_devices", None)
        out["interface_class"] = type(iface).__name__
        out["include_dram"] = getattr(iface, "emi_include_dram", None)
        if devices is None:
            out["probe_error"] = "attribut _devices introuvable (API CodeCarbon modifiée ?)"
            return out
        sel, monitored = [], []
        for path, names, idxs in devices:
            monitored.extend(names)
            sel.extend({"device_path": path, "channel_index": i, "channel": names[i]} for i in idxs)
        out.update({"selected_channels": sel, "selected_channel_names": [s["channel"] for s in sel],
                    "channels_on_monitored_devices": monitored, "probe_ok": True})
        return out

    def describe(self, n_tasks, n_cpu_nonpositive, expected_channel):
        try:
            self.t._ensure_hardware_ready()
        except Exception:                                   # noqa: BLE001
            pass
        info: Dict[str, Any] = {"backend": self.name, "simulated": False,
                                "introspection": "attributs privés de CodeCarbon (stabilité non garantie)",
                                "raw_samples_exposed": False,
                                "raw_log_content": "enregistrements aux bornes des tâches uniquement (valeurs de stop_task + horodatages)"}
        try:
            import importlib.metadata as imd
            info["codecarbon_version"] = imd.version("codecarbon")
        except Exception:                                   # noqa: BLE001
            info["codecarbon_version"] = None
        mode = None
        for h in getattr(self.t, "_hardware", []):
            if type(h).__name__ == "CPU":
                mode = getattr(h, "_mode", None)
                info["cpu_tdp_w"] = getattr(h, "_tdp", None)
        probe = self.probe_emi() if str(mode).lower() in ("windows_emi", "intel_rapl") else {"probe_ok": False}
        info["emi_probe"] = probe
        info.update(classify_sources(mode, probe, n_tasks, n_cpu_nonpositive, expected_channel))
        return info

    def close(self):
        try:
            self.t.stop()
        finally:
            CodeCarbonBackend._ACTIVE = None


class FakeBackend(Backend):
    """Backend SIMULÉ pour les tests : énergie = puissance(nom de tâche) x durée réelle. N'est JAMAIS une mesure matérielle :
    classify_sources(simulated=True) -> 'simulated_not_hardware'."""
    name = "fake_simulated"
    simulated = True

    def __init__(self, power_fn: Optional[Callable[[str], Tuple[float, float]]] = None, duration_scale: float = 1.0,
                 duration_offset_s: float = 0.0, cpu_zero: bool = False):
        self.power_fn = power_fn or (lambda name: (5.0, 1.0))
        self.duration_scale, self.duration_offset_s, self.cpu_zero = duration_scale, duration_offset_s, cpu_zero
        self._name, self._t0 = None, None
        self.calls: List[str] = []

    def start_task(self, name):
        if self._name is not None:
            raise EnergyError("tâche déjà active (simulé)")
        self._name, self._t0 = name, time.perf_counter()
        self.calls.append("start:" + name)

    def stop_task(self):
        d = (time.perf_counter() - self._t0) * self.duration_scale + self.duration_offset_s
        cpu_w, ram_w = self.power_fn(self._name)
        if self.cpu_zero:
            cpu_w = 0.0
        self.calls.append("stop:" + self._name)
        self._name = None
        return {"energy_total_j": (cpu_w + ram_w) * d, "energy_cpu_j": cpu_w * d, "energy_ram_j": ram_w * d,
                "energy_gpu_j": 0.0, "duration_s": d, "cpu_power_w_last": cpu_w, "ram_power_w_last": ram_w}

    def describe(self, n_tasks, n_cpu_nonpositive, expected_channel):
        info = {"backend": self.name, "simulated": True, "raw_samples_exposed": False,
                "raw_log_content": "enregistrements aux bornes des tâches (simulés)"}
        info.update(classify_sources("simulated", {}, n_tasks, n_cpu_nonpositive, expected_channel, simulated=True))
        return info


# =====================================================================================
# 7. Validation
# =====================================================================================
def _chk(name: str, status: str, detail: str, value: Any = None) -> Dict[str, Any]:
    return {"check": name, "status": status, "detail": detail, "value": value}


def has_fail(checks: Sequence[Mapping[str, Any]]) -> bool:
    return any(c["status"] == "fail" for c in checks)


def validate_task(rec: Mapping[str, Any], cfg: MeterConfig) -> List[Dict[str, Any]]:
    """Contrôles d'UNE tâche : valeurs finies, ordre des horodatages, cohérence des durées, fenêtre assez longue, énergie CPU > 0,
    total = somme des composants. Les statuts sont ok / warn / fail / info."""
    c: List[Dict[str, Any]] = []
    es = [rec.get(f"energy_{KEY[k]}_j") for k in ("cpu", "ram", "total")]
    c.append(_chk("energies_finies", "ok" if all(isfin(e) for e in es) else "fail",
                  "énergies CPU/RAM/total finies" if all(isfin(e) for e in es) else f"non finies : {es}"))
    win = rec.get("window_s", NAN)
    c.append(_chk("fenetre_valide", "ok" if isfin(win) and win > 0 else "fail", f"fenêtre = {win}", win))
    mono_ok = isfin(rec.get("t_start_mono_ns")) and isfin(rec.get("t_stop_mono_ns")) and rec["t_stop_mono_ns"] > rec["t_start_mono_ns"]
    wall_ok = isfin(rec.get("t_start_wall_ns")) and isfin(rec.get("t_stop_wall_ns")) and rec["t_stop_wall_ns"] >= rec["t_start_wall_ns"]
    c.append(_chk("horodatages_ordonnes", "ok" if mono_ok and wall_ok else "fail",
                  f"monotonique ok={mono_ok}, mural ok={wall_ok}"))
    gap = rec.get("window_gap_pct", NAN)
    if not isfin(gap):
        c.append(_chk("coherence_durees", "fail", "écart fenêtre/durée réelle non calculable"))
    else:
        c.append(_chk("coherence_durees", "ok" if gap <= cfg.duration_tolerance_pct else "warn",
                      f"écart {gap:.3f} % (tolérance {cfg.duration_tolerance_pct} %)", gap))
    if isfin(win) and win > 0:
        short = win < cfg.min_window_s
        c.append(_chk("fenetre_ge_periodes", "fail" if short else "ok",
                      f"fenêtre {win:.4g} s {'<' if short else '>='} {cfg.min_window_periods:g} x {cfg.measure_power_secs:g} s "
                      f"= {cfg.min_window_s:g} s", win / cfg.measure_power_secs))
    cpu = rec.get("energy_cpu_j", NAN)
    if isfin(cpu):
        c.append(_chk("energie_cpu_positive", "ok" if cpu > 0 else "warn", f"énergie CPU = {cpu:.6g} J", cpu))
    tot, parts = rec.get("energy_total_hybrid_j", NAN), [rec.get(f"energy_{KEY[k]}_j", NAN) for k in ("cpu", "ram", "gpu")]
    if isfin(tot) and all(isfin(p) for p in parts):
        res = tot - sum(parts)
        c.append(_chk("total_egal_somme_composants", "ok" if abs(res) <= 1e-6 * max(1.0, abs(tot)) else "fail",
                      f"résidu = {res:.3g} J", res))
    if rec.get("error"):
        c.append(_chk("pas_d_exception", "fail", str(rec["error"])))
    return c


def validate_idle(baseline: Mapping[str, Any], cfg: MeterConfig, drift: Optional[Mapping[str, float]] = None) -> List[Dict[str, Any]]:
    c: List[Dict[str, Any]] = []
    n = baseline.get("n_records", 0)
    c.append(_chk("idle_nombre_mesures", "ok" if n >= cfg.idle_min_records else "warn",
                  f"{n} mesure(s) idle (minimum {cfg.idle_min_records})", n))
    for comp in ("cpu", "total"):
        cv = baseline.get(comp, {}).get("cv", NAN)
        if isfin(cv):
            c.append(_chk(f"idle_cv_{comp}", "ok" if cv <= cfg.idle_cv_max else "warn",
                          f"CV idle {comp} = {cv:.4f} (max {cfg.idle_cv_max})", cv))
        else:
            c.append(_chk(f"idle_cv_{comp}", "info", "CV non défini (< 2 mesures)"))
    for comp, v in (drift or {}).items():
        if isfin(v):
            c.append(_chk(f"idle_derive_{comp}", "ok" if abs(v) <= cfg.idle_drift_max_rel else "warn",
                          f"dérive relative avant/après = {v:+.4f} (max {cfg.idle_drift_max_rel})", v))
    return c


def validate_load_vs_idle(task: Mapping[str, Any], baseline: Mapping[str, Any], cfg: MeterConfig) -> List[Dict[str, Any]]:
    """Critère de CONTRÔLE configurable (pas une vérité universelle) : P_charge / P_idle. Valeurs réelles toujours enregistrées."""
    out: List[Dict[str, Any]] = []
    for comp in ("cpu", "total"):
        p, pi = task.get(f"mean_power_{KEY[comp]}_w", NAN), baseline.get(comp, {}).get("mean_w", NAN)
        ratio = p / pi if isfin(p) and isfin(pi) and pi > 0 else NAN
        if not cfg.load_idle_ratio_check:
            out.append(_chk(f"charge_sur_idle_{comp}", "info", f"contrôle désactivé ; rapport = {ratio}", ratio))
        elif not isfin(ratio):
            out.append(_chk(f"charge_sur_idle_{comp}", "info", "rapport non calculable", ratio))
        else:
            ok = ratio >= cfg.load_idle_ratio_min
            out.append(_chk(f"charge_sur_idle_{comp}", "ok" if ok else cfg.load_idle_ratio_severity,
                            f"P_charge/P_idle = {ratio:.4f} (seuil configurable {cfg.load_idle_ratio_min})", ratio))
    return out


# =====================================================================================
# 8. Gestionnaire de mesure
# =====================================================================================
class EnergyMeter:
    """Gestionnaire : tâches longues, boucles (tâches courtes), idle avant/après, brut/net, validation, journal brut.
    Une seule tâche active à la fois ; initialisation et warmup toujours HORS tâche mesurée."""

    def __init__(self, backend: Backend, config: Optional[MeterConfig] = None, raw_log_path=None):
        self.backend = backend
        self.cfg = config or MeterConfig()
        self.tasks: List[Dict[str, Any]] = []
        self.idles: List[Dict[str, Any]] = []
        self.cpu_nonpositive_tasks: List[str] = []
        self._active: Optional[str] = None
        self.raw = RawLog(raw_log_path) if raw_log_path else None
        if self.raw:
            self.raw.append({"event": "meter_open", "module_version": MODULE_VERSION, "backend": backend.name,
                             "simulated": backend.simulated, "t_wall_ns": time.time_ns(), "t_mono_ns": time.monotonic_ns(),
                             "config": dataclasses.asdict(self.cfg)})

    # ----------------------------------------------------------------- tâche
    @contextlib.contextmanager
    def task(self, name: str, kind: str = "task", n_items: int = 1):
        if self._active is not None:
            raise EnergyError(f"chevauchement de tâches : '{self._active}' active, '{name}' demandée")
        self._active = name
        holder: Dict[str, Any] = {"name": name}
        t_call0 = time.monotonic_ns()
        try:
            self.backend.start_task(name)
        except BaseException:
            self._active = None
            raise
        s_mono, s_wall = time.monotonic_ns(), time.time_ns()
        err = None
        try:
            yield holder
        except BaseException as e:                          # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            raise
        finally:
            e_mono, e_wall = time.monotonic_ns(), time.time_ns()
            try:
                raw = self.backend.stop_task()
            finally:
                self._active = None
            t_call1 = time.monotonic_ns()
            rec = self._build_record(name, kind, int(holder.get("n_items", n_items)), raw, s_mono, s_wall, e_mono, e_wall,
                                     t_call1 - t_call0, err)
            holder.update(rec)
            self.tasks.append(rec)
            if not (isfin(rec["energy_cpu_j"]) and rec["energy_cpu_j"] > 0):
                self.cpu_nonpositive_tasks.append(name)
            if self.raw:
                self.raw.append({"event": "task", **{k: v for k, v in rec.items() if k != "checks"},
                                 "backend_raw": json_safe(raw)})

    def _build_record(self, name, kind, n_items, raw, s_mono, s_wall, e_mono, e_wall, outer_ns, err):
        wall_s = (e_mono - s_mono) / 1e9
        cc = raw.get("duration_s", NAN)
        win, src = choose_window(cc, wall_s)
        gap_s, gap_pct = window_gap(wall_s, win)
        rec: Dict[str, Any] = {
            "task": name, "kind": kind, "n_items": int(n_items),
            "t_start_mono_ns": s_mono, "t_stop_mono_ns": e_mono, "t_start_wall_ns": s_wall, "t_stop_wall_ns": e_wall,
            "wall_s": wall_s, "backend_duration_s": cc, "window_s": win, "window_source": src,
            "window_gap_s": gap_s, "window_gap_pct": gap_pct, "outer_s": outer_ns / 1e9,
            "tracker_overhead_s": outer_ns / 1e9 - wall_s,
            "window_over_sampling": win / self.cfg.measure_power_secs if isfin(win) else NAN,
            "window_too_short": bool(not isfin(win) or win < self.cfg.min_window_s),
            "energy_cpu_j": raw.get("energy_cpu_j", NAN), "energy_ram_modelled_j": raw.get("energy_ram_j", NAN),
            "energy_gpu_j": raw.get("energy_gpu_j", 0.0), "energy_total_hybrid_j": raw.get("energy_total_j", NAN),
            "cpu_power_w_last_sample": raw.get("cpu_power_w_last", NAN),
            "ram_power_w_last_sample": raw.get("ram_power_w_last", NAN), "error": err}
        for c in COMPS:
            rec[f"mean_power_{KEY[c]}_w"] = mean_power_w(rec[f"energy_{KEY[c]}_j"], win)
        rec["checks"] = validate_task(rec, self.cfg)
        rec["valid"] = not has_fail(rec["checks"])
        return rec

    # ----------------------------------------------------------------- idle
    def measure_idle(self, seconds: float, label: str = "idle", allow_short: bool = False) -> Dict[str, Any]:
        if seconds < self.cfg.min_window_s and not allow_short:
            raise WindowTooShortError(f"idle {seconds:g} s < {self.cfg.min_window_periods:g} x {self.cfg.measure_power_secs:g} s")
        with self.task(label, kind="idle") as h:
            time.sleep(seconds)
        rec = dict(h)
        rec["requested_s"] = seconds
        self.idles.append(rec)
        return rec

    # ----------------------------------------------------------------- blocs
    def _block(self, name, run: Callable[[], Any], kind, n_items_fn, idle_before_s, idle_after_s, baseline_override):
        before = self.measure_idle(idle_before_s, f"{name}::idle_before") if idle_before_s and idle_before_s > 0 else None
        value = None
        with self.task(name, kind=kind) as h:
            value = run()
            h["n_items"] = int(n_items_fn(value))      # lu par task() à la sortie : le journal brut est cohérent
        task_rec = dict(h)
        after = self.measure_idle(idle_after_s, f"{name}::idle_after") if idle_after_s and idle_after_s > 0 else None
        idle_recs = [r for r in (before, after) if r is not None]
        base = baseline_override if baseline_override is not None else idle_baseline(idle_recs)
        drift = idle_drift(before, after) if before and after else {}
        energy = compute_energy(task_rec, base, sampling_period_s=self.cfg.measure_power_secs,
                                boundary_enabled=self.cfg.boundary_enabled,
                                boundary_half_width_periods=self.cfg.boundary_half_width_periods,
                                significance_k=self.cfg.significance_k, n_items=task_rec["n_items"])
        checks = list(task_rec["checks"])
        checks += validate_idle(base, self.cfg, drift)
        checks += validate_load_vs_idle(task_rec, base, self.cfg)
        if not energy["net_consistent"]:
            checks.append(_chk("net_coherent", "fail", f"écart net composants/direct = {energy['net_component_sum_difference_j']}"))
        block = {"name": name, "kind": kind, "task": {k: v for k, v in task_rec.items() if k != "checks"},
                 "idle_before": before, "idle_after": after, "baseline": base, "idle_drift_relative": drift,
                 "energy": energy, "checks": checks, "valid": not has_fail(checks),
                 "baseline_source": "external" if baseline_override is not None else "bracketing_before_after"}
        if self.raw:
            self.raw.append({"event": "block", **json_safe({k: v for k, v in block.items() if k not in ("idle_before", "idle_after")})})
        return value, block

    def measure_long(self, name: str, fn: Callable[[], Any], *, n_items: int = 1, idle_before_s: float = 0.0,
                     idle_after_s: float = 0.0, baseline: Optional[Mapping[str, Any]] = None):
        """Tâche longue : UNE exécution de fn() dans une tâche, encadrée (option) par des idle. Retourne (valeur, bloc)."""
        value, block = self._block(name, fn, "long", lambda v: n_items, idle_before_s, idle_after_s, baseline)
        block["statement"] = "énergie d'une exécution unique de la tâche (fenêtre >= périodes minimales contrôlée)"
        return value, block

    def measure_loop(self, name: str, fn: Callable[[], Optional[int]], *, min_seconds: float, min_iterations: int = 1,
                     warmup_iterations: int = 0, items_per_call: int = 1, idle_before_s: float = 0.0,
                     idle_after_s: float = 0.0, baseline: Optional[Mapping[str, Any]] = None, allow_short: bool = False):
        """Tâche courte mesurée PAR BOUCLE. fn() traite des éléments et peut retourner leur nombre (sinon items_per_call).
        Le warmup est exécuté HORS tâche mesurée. Rapporte : énergie totale de la boucle, N éléments, énergie MOYENNE par
        élément, temps moyen par élément. Ce n'est PAS l'énergie d'un élément isolé."""
        if min_seconds < self.cfg.min_window_s and not allow_short:
            raise WindowTooShortError(f"min_seconds={min_seconds:g} < {self.cfg.min_window_periods:g} x "
                                      f"{self.cfg.measure_power_secs:g} s = {self.cfg.min_window_s:g} s : fenêtre trop courte")
        w0, w0w = time.monotonic_ns(), time.time_ns()
        for _ in range(int(warmup_iterations)):
            fn()
        warm = {"warmup_iterations": int(warmup_iterations), "warmup_s": (time.monotonic_ns() - w0) / 1e9,
                "warmup_t_start_wall_ns": w0w}
        counters = {"iters": 0, "items": 0}

        def run():
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < min_seconds or counters["iters"] < min_iterations:
                r = fn()
                counters["iters"] += 1
                counters["items"] += int(r) if isinstance(r, int) and not isinstance(r, bool) else int(items_per_call)
            return dict(counters)
        value, block = self._block(name, run, "loop", lambda v: max(v["items"], 1), idle_before_s, idle_after_s, baseline)
        n = block["task"]["n_items"]
        block["loop"] = {"n_iterations": counters["iters"], "n_items": n, "min_seconds_requested": min_seconds,
                         "time_per_item_s": block["task"]["wall_s"] / n, **warm}
        block["statement"] = ("énergie TOTALE d'une boucle ; la valeur par élément est une MOYENNE sur la boucle, "
                              "pas la mesure d'un élément individuel")
        if self.raw:
            self.raw.append({"event": "loop_summary", "name": name, **json_safe(block["loop"])})
        return value, block

    # ----------------------------------------------------------------- bilan
    def method_info(self) -> Dict[str, Any]:
        return self.backend.describe(len(self.tasks), len(self.cpu_nonpositive_tasks), self.cfg.expected_channel)

    def summary(self) -> Dict[str, Any]:
        return {"module_version": MODULE_VERSION, "config": dataclasses.asdict(self.cfg), "method_info": self.method_info(),
                "n_tasks": len(self.tasks), "n_tasks_invalid": sum(1 for t in self.tasks if not t["valid"]),
                "n_idle": len(self.idles), "idle_baseline_all": idle_baseline(self.idles)}

    def close(self) -> None:
        try:
            if self.raw:
                self.raw.append({"event": "meter_close", "t_wall_ns": time.time_ns(), "t_mono_ns": time.monotonic_ns(),
                                 "n_tasks": len(self.tasks)})
                self.raw.close()
        finally:
            self.backend.close()


def new_codecarbon_meter(cfg: MeterConfig, raw_log_path=None) -> EnergyMeter:
    """Fabrique : EnergyMeter sur CodeCarbon (import paresseux)."""
    return EnergyMeter(CodeCarbonBackend(cfg), cfg, raw_log_path)


# =====================================================================================
# 9. Sonde de l'API CodeCarbon (à lancer sur Windows ; non exécutée par les tests)
# =====================================================================================
def probe_codecarbon_api(measure_power_secs: float = 1.0, tracking_mode: str = "machine") -> Dict[str, Any]:
    """Liste ce que l'installation RÉELLE de CodeCarbon expose (noms et types seulement, aucune hypothèse) afin de décider si des
    échantillons bruts RAPL/EMI sont accessibles. Crée puis arrête un tracker."""
    out: Dict[str, Any] = {"module_version": MODULE_VERSION, "python": sys.version, "platform": sys.platform}
    try:
        import importlib.metadata as imd
        out["codecarbon_version"] = imd.version("codecarbon")
        from codecarbon import OfflineEmissionsTracker
    except Exception as e:                                  # noqa: BLE001
        out["available"] = False
        out["error"] = repr(e)
        return out
    out["available"] = True
    pat = re.compile(r"energy|power|raw|sample|counter|channel|device|measure", re.I)
    t = OfflineEmissionsTracker(country_iso_code="MAR", measure_power_secs=measure_power_secs, save_to_file=False,
                                log_level="error", tracking_mode=tracking_mode, rapl_include_dram=False)
    try:
        t.start()
        try:
            t._ensure_hardware_ready()
        except Exception as e:                              # noqa: BLE001
            out["ensure_hardware_ready_error"] = repr(e)
        out["tracker_members_matching"] = {n: type(getattr(t, n, None)).__name__ for n in dir(t) if pat.search(n)}
        hw = []
        for h in getattr(t, "_hardware", []):
            item = {"class": type(h).__name__, "mode": getattr(h, "_mode", None),
                    "members_matching": {n: type(getattr(h, n, None)).__name__ for n in dir(h) if pat.search(n)}}
            iface = getattr(h, "_intel_interface", None)
            if iface is not None:
                item["interface_class"] = type(iface).__name__
                item["interface_members_matching"] = {n: type(getattr(iface, n, None)).__name__ for n in dir(iface) if pat.search(n)}
            hw.append(item)
        out["hardware"] = hw
        out["conclusion_rule"] = ("Des échantillons bruts ne sont considérés exposés que si un membre listé ci-dessus contient "
                                  "réellement une série horodatée ; cette sonde ne le démontre pas : à inspecter manuellement.")
    finally:
        try:
            t.stop()
        except Exception as e:                              # noqa: BLE001
            out["stop_error"] = repr(e)
    return out


if __name__ == "__main__":                                   # python energy_profiling_v52.py  -> JSON de la sonde sur stdout
    print(dumps_strict(probe_codecarbon_api()))
