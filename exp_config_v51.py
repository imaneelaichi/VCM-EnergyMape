# -*- coding: utf-8 -*-
"""
exp_config_v51.py -- lecture et validation de experiment_config_v51.yaml pour VCM-EnergyMap exp.3 (V5.1 = V5 + V4.1).

Ce module ne lance AUCUNE mesure et ne modifie NI le codec, NI les poids. Il :
  * charge le YAML, fusionne `defaults` + expérience, résout chemins / nom de dossier ;
  * vérifie la compatibilité des paramètres (GOP, structure I/P attendue, checkpoint, résolution, images, détecteur,
    paramètres d'énergie, dossier de sortie) ;
  * inspecte les checkpoints SANS rien supposer sur leur signification (pas de notion de "niveau de qualité") ;
  * prépare les images d'entrée à la résolution demandée (redimensionnement des ENTRÉES, cache dédié, sans écrasement) ;
  * construit l'objet `a` (argparse.Namespace) attendu par le moteur de mesure.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path

OK, FAIL, WARN, INFO, TODO = "OK", "FAIL", "WARN", "INFO", "NON_VÉRIFIÉ"
IDLE_MIN_RATIO = 10
RESAMPLE = {"nearest": "NEAREST", "bilinear": "BILINEAR", "bicubic": "BICUBIC", "lanczos": "LANCZOS"}


class ConfigError(Exception):
    pass


# ------------------------------------------------------------------------------------ utilitaires
def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def canon_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_resolution(s):
    m = re.fullmatch(r"\s*(\d+)\s*[x×X]\s*(\d+)\s*", str(s))
    if not m:
        raise ConfigError(f"résolution invalide '{s}' (attendu : 768x432)")
    return int(m.group(1)), int(m.group(2))


def ckpt_id_from_name(name):
    m = re.search(r"(\d+)$", str(name))
    if not m:
        raise ConfigError(f"impossible de déduire l'identifiant numérique de '{name}' (attendu : base_1)")
    return int(m.group(1))


def load_config(path):
    try:
        import yaml
    except ImportError as e:
        raise ConfigError("PyYAML absent : installez-le avec  python -m pip install pyyaml") from e
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"fichier de configuration introuvable : {p}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError("le YAML doit contenir un dictionnaire à la racine")
    for k in ("paths", "allowed", "defaults", "experiments"):
        if k not in cfg:
            raise ConfigError(f"clé obligatoire absente du YAML : '{k}'")
    ids = [e.get("id") for e in cfg["experiments"]]
    if len(set(ids)) != len(ids) or any(not i for i in ids):
        raise ConfigError(f"identifiants d'expériences manquants ou dupliqués : {ids}")
    cfg["_file"] = str(p.resolve())
    cfg["_sha256"] = sha256_file(p)
    return cfg


def list_ids(cfg, only_enabled=False):
    return [e["id"] for e in cfg["experiments"] if not only_enabled or e.get("enabled", True)]


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s))


# ------------------------------------------------------------------------------------ résolution
def resolve(cfg, exp_id, project_root):
    exps = {e["id"]: e for e in cfg["experiments"]}
    if exp_id not in exps:
        raise ConfigError(f"expérience '{exp_id}' absente du YAML (disponibles : {list(exps)})")
    raw = exps[exp_id]
    r = deep_merge(cfg["defaults"], {k: v for k, v in raw.items() if k not in ("id", "enabled")})
    root = Path(project_root)
    P = cfg["paths"]
    W, H = parse_resolution(r["resolution"])
    ck = str(r["checkpoint"])
    count, reps, gop = int(r["frames"]["count"]), int(r["repeats"]), int(r["gop"])
    dirname = safe_name(f"{exp_id}_gop{gop}_{W}x{H}_{ck}_{r['video']['name']}_n{count}_r{reps}")
    res_root = Path(P["results_root"]);  res_root = res_root if res_root.is_absolute() else root / res_root
    ck_dir = Path(P["checkpoints_dir"]); ck_dir = ck_dir if ck_dir.is_absolute() else root / ck_dir
    frames_dir = Path(r["video"]["frames_dir"]); frames_dir = frames_dir if frames_dir.is_absolute() else root / frames_dir
    cache = Path(P.get("inputs_cache", "inputs_cache")); cache = cache if cache.is_absolute() else root / cache
    R = {"id": exp_id, "gop": gop, "resolution": f"{W}x{H}", "W": W, "H": H, "checkpoint": ck,
         "ckpt_id": ckpt_id_from_name(ck), "frames_count": count, "repeats": reps,
         "video": r["video"], "codec": r["codec"], "resize": r["resize"], "detector": r["detector"],
         "energy": r["energy"], "measurement": r["measurement"], "validation": r["validation"],
         "estimate": r["estimate"], "limits": r["limits"], "allowed": cfg["allowed"],
         "paths": {"project_root": str(root), "results_root": str(res_root), "results_group": P["results_group"],
                   "checkpoint_file": str(ck_dir / P["checkpoint_file_pattern"].format(checkpoint=ck)),
                   "checkpoints_dir": str(ck_dir), "frames_dir": str(frames_dir), "inputs_cache": str(cache),
                   "dirname": dirname, "out_dir": str(res_root / P["results_group"] / dirname)},
         "batch": cfg.get("batch", {}), "config_file": cfg["_file"], "config_sha256": cfg["_sha256"],
         "experiment_entry": raw}
    R["detector_hash"] = canon_hash(R["detector"])
    # cadence de la vidéo (bit/s) : video.fps ; absente/null -> 30 images/s SUPPOSÉES, signalé partout (fps_assumed)
    fps_raw = r["video"].get("fps")
    try:
        R["fps"] = float(fps_raw) if fps_raw not in (None, "", 0) else 30.0
    except (TypeError, ValueError) as e:
        raise ConfigError(f"video.fps invalide : {fps_raw!r}") from e
    R["fps_assumed"] = fps_raw in (None, "", 0)
    return R


def frame_path(directory, prefix, idx):
    return os.path.join(directory, f"{prefix}{idx:03d}.png")


# ------------------------------------------------------------------------------------ checkpoints
def inspect_checkpoint(path, with_sha=True):
    """Décrit un checkpoint SANS supposer son contenu : clés, champs scalaires de type configuration, tenseurs,
    signature (noms + formes). N'interprète pas ces champs comme un 'niveau de qualité'."""
    p = Path(path)
    rep = {"path": str(p), "exists": p.exists(), "loaded": False}
    if not p.exists():
        return rep
    rep["size_bytes"] = p.stat().st_size
    if with_sha:
        rep["sha256"] = sha256_file(p)
    try:
        import torch
    except ImportError:
        rep["error"] = "torch absent : inspection impossible"
        return rep
    try:
        try:
            obj = torch.load(str(p), map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(str(p), map_location="cpu")
    except Exception as e:                                    # noqa: BLE001
        rep["error"] = f"torch.load a échoué : {e!r}"
        return rep
    rep["loaded"] = True
    rep["python_type"] = type(obj).__name__

    def scal(v):
        return isinstance(v, (int, float, str, bool)) or v is None

    cfgfields, state = {}, None
    if isinstance(obj, dict):
        rep["top_level_keys"] = [str(k) for k in list(obj.keys())[:60]]
        for k, v in obj.items():
            if scal(v):
                cfgfields[str(k)] = v
            elif isinstance(v, (list, tuple)) and len(v) <= 20 and all(scal(x) for x in v):
                cfgfields[str(k)] = list(v)
            elif isinstance(v, dict) and 0 < len(v) <= 60 and all(scal(x) for x in v.values()):
                for kk, vv in v.items():
                    cfgfields[f"{k}.{kk}"] = vv
        for key in ("state_dict", "model", "net", "network", "params"):
            if isinstance(obj.get(key), dict):
                state = obj[key]
                rep["state_dict_key"] = key
                break
        if state is None and obj and all(hasattr(v, "shape") for v in obj.values()):
            state, rep["state_dict_key"] = obj, "<racine>"
    rep["config_like_fields"] = cfgfields
    if state:
        items = [(str(k), tuple(int(d) for d in v.shape)) for k, v in state.items() if hasattr(v, "shape")]
        rep["n_tensors"] = len(items)
        rep["n_parameters"] = int(sum(math.prod(s) if s else 1 for _, s in items))
        rep["dtypes"] = sorted({str(v.dtype) for v in state.values() if hasattr(v, "dtype")})
        rep["signature_sha256"] = canon_hash(sorted(items))
        pref = {}
        for k, _ in items:
            pref[k.split(".")[0]] = pref.get(k.split(".")[0], 0) + 1
        rep["top_modules"] = pref
    return rep


def compare_checkpoints(reports):
    """reports : {nom: rapport}. Compare signatures (architecture/formes) et champs de configuration."""
    out = {"names": list(reports), "same_signature": None, "signature_groups": {}, "differing_config_fields": {}}
    sigs = {n: r.get("signature_sha256") for n, r in reports.items() if r.get("signature_sha256")}
    for n, s in sigs.items():
        out["signature_groups"].setdefault(s, []).append(n)
    if len(sigs) == len(reports) and len(reports) > 1:
        out["same_signature"] = len(out["signature_groups"]) == 1
    keys = set()
    for r in reports.values():
        keys |= set((r.get("config_like_fields") or {}).keys())
    for k in sorted(keys):
        vals = {n: (r.get("config_like_fields") or {}).get(k, "<absent>") for n, r in reports.items()}
        if len({json.dumps(v, default=str) for v in vals.values()}) > 1:
            out["differing_config_fields"][k] = vals
    return out


# ------------------------------------------------------------------------------------ entrées
def source_frame_files(R):
    v = R["video"]
    return [frame_path(R["paths"]["frames_dir"], v["prefix"], v["start_index"] + i) for i in range(R["frames_count"])]


def input_plan(R):
    """Décide quel dossier d'images alimente le codec : source telle quelle, ou cache redimensionné."""
    from PIL import Image
    files = source_frame_files(R)
    present = [f for f in files if os.path.exists(f)]
    plan = {"n_expected": len(files), "n_present": len(present), "source_dir": R["paths"]["frames_dir"],
            "source_sizes": {}, "action": None, "inputs_dir": None, "method": R["resize"]["method"]}
    if not present:
        return plan
    for f in present:
        with Image.open(f) as im:
            plan["source_sizes"].setdefault(f"{im.size[0]}x{im.size[1]}", []).append(os.path.basename(f))
    if len(plan["source_sizes"]) == 1:
        sw, sh = map(int, next(iter(plan["source_sizes"])).split("x"))
        plan["source_resolution"] = f"{sw}x{sh}"
        if (sw, sh) == (R["W"], R["H"]):
            plan.update(action="use_source", inputs_dir=R["paths"]["frames_dir"])
        else:
            plan.update(action="resize", inputs_dir=str(Path(R["paths"]["inputs_cache"]) /
                                                        safe_name(f"{R['video']['name']}_{R['W']}x{R['H']}_{R['resize']['method']}")))
    return plan


def prepare_inputs(R, plan=None, execute=True):
    """Crée (si besoin) les images redimensionnées dans le cache dédié. Ne touche jamais aux images sources et
    n'écrase jamais un fichier du cache (un fichier de mauvaise taille = erreur)."""
    from PIL import Image
    plan = plan or input_plan(R)
    info = dict(plan)
    info["created"], info["reused"] = [], []
    if plan["action"] == "resize" and execute:
        res = getattr(Image, "Resampling", Image)
        method = getattr(res, RESAMPLE[R["resize"]["method"]])
        out = Path(plan["inputs_dir"])
        out.mkdir(parents=True, exist_ok=True)
        for f in source_frame_files(R):
            dst = out / os.path.basename(f)
            if dst.exists():
                with Image.open(dst) as im:
                    if im.size != (R["W"], R["H"]):
                        raise ConfigError(f"{dst} existe avec une taille {im.size} != {(R['W'], R['H'])} : "
                                          f"supprimez-le manuellement (jamais écrasé automatiquement)")
                info["reused"].append(dst.name)
                continue
            with Image.open(f) as im:
                im.convert("RGB").resize((R["W"], R["H"]), method).save(dst)
            info["created"].append(dst.name)
    if plan["inputs_dir"]:
        info["inputs_sha256"] = {os.path.basename(f): (sha256_file(frame_path(plan["inputs_dir"], R["video"]["prefix"],
                                                                              R["video"]["start_index"] + i))
                                                       if os.path.exists(frame_path(plan["inputs_dir"], R["video"]["prefix"],
                                                                                    R["video"]["start_index"] + i)) else None)
                                 for i, f in enumerate(source_frame_files(R))}
    return info


# ------------------------------------------------------------------------------------ estimation
def estimate_params(R):
    e = R["estimate"]
    f = 1.0
    if e.get("scale_with_pixels", True):
        rw, rh = parse_resolution(e["reference_resolution"])
        f = (R["W"] * R["H"]) / (rw * rh)
    return {k: float(e[k]) * f for k in ("enc_I_s", "enc_P_s", "dec_I_s", "dec_P_s")}, f


def expected_types(n, gop):
    return ["I" if i % gop == 0 else "P" for i in range(n)]


# ------------------------------------------------------------------------------------ vérifications
def static_checks(R, deep=False, ckpt_cache=None):
    """Retourne une liste de {check, status, detail}. deep=True : inspection torch des checkpoints (plus lourd)."""
    out = []

    def add(check, status, detail):
        out.append({"check": check, "status": status, "detail": detail})

    n, gop = R["frames_count"], R["gop"]
    # --- paramètres autorisés
    for key, val in (("gop", gop), ("resolution", R["resolution"]), ("checkpoint", R["checkpoint"])):
        ok = val in R["allowed"][key]
        add(f"valeur autorisée : {key}", OK if ok else FAIL, f"{val} ; autorisé : {R['allowed'][key]}")
    if n < 1 or R["repeats"] < 1:
        add("nombre d'images / répétitions", FAIL, f"images={n}, répétitions={R['repeats']}")
    # --- 1/2. GOP et structure I/P attendue
    types = expected_types(n, gop)
    n_i = types.count("I")
    add("GOP : structure I/P attendue", OK if (gop >= 2 and n - n_i >= 1) else FAIL,
        f"GOP={gop}, {n} images -> {''.join(types)} ({n_i} I + {n - n_i} P)")
    if n_i == 1:
        add("une seule image I par passe", INFO, "la variabilité de l'image I vient uniquement des répétitions")
    add("GOP accepté par le codec ; structure I/P réellement produite", TODO,
        "vérifié par --probe (encodage/décodage réels, type observé via les étapes du profiler) puis, pendant la mesure, "
        "image par image (arrêt si le type observé diffère du type attendu)")
    # --- 5. images disponibles
    v = R["video"]
    plan = input_plan(R)
    if plan["n_present"] < plan["n_expected"]:
        missing = [os.path.basename(f) for f in source_frame_files(R) if not os.path.exists(f)]
        add("images disponibles", FAIL, f"{plan['n_present']}/{plan['n_expected']} présentes dans {plan['source_dir']} ; "
                                        f"manquantes : {missing[:8]}")
    else:
        add("images disponibles", OK, f"{plan['n_present']} images {v['prefix']}{v['start_index']:03d}.png.. dans {plan['source_dir']}")
    # --- cadence (débits en bit/s)
    if not R["fps"] > 0:
        add("cadence fps", FAIL, f"{R['fps']}")
    else:
        add("cadence fps", WARN if R["fps_assumed"] else OK,
            f"{R['fps']:g} images/s " + ("SUPPOSÉE (video.fps non renseigné) : les bit/s en dépendent, pas les bpp"
                                         if R["fps_assumed"] else "(video.fps du YAML)"))
    # --- 4. résolution
    if len(plan["source_sizes"]) > 1:
        add("résolution source homogène", FAIL, f"tailles différentes : { {k: len(x) for k, x in plan['source_sizes'].items()} }")
    elif plan.get("source_resolution"):
        sw, sh = map(int, plan["source_resolution"].split("x"))
        add("résolution source", INFO, plan["source_resolution"])
        if plan["action"] == "use_source":
            add("résolution demandée", OK, f"{R['resolution']} = résolution source : images utilisées telles quelles")
        else:
            asp_ok = abs(sw / sh - R["W"] / R["H"]) < 2e-3
            up = R["W"] > sw or R["H"] > sh
            add("rapport d'aspect", OK if (asp_ok or R["resize"]["allow_aspect_change"]) else FAIL,
                f"source {sw}x{sh} ({sw / sh:.4f}) -> cible {R['W']}x{R['H']} ({R['W'] / R['H']:.4f})")
            add("agrandissement", OK if (not up or R["resize"]["allow_upscale"]) else FAIL,
                "cible plus grande que la source" if up else "réduction uniquement")
            if R["resize"]["method"] not in RESAMPLE:
                add("méthode de redimensionnement", FAIL, f"{R['resize']['method']} (autorisées : {list(RESAMPLE)})")
            add("résolution demandée", OK,
                f"redimensionnement {R['resize']['method']} {sw}x{sh} -> {R['W']}x{R['H']} dans {plan['inputs_dir']} "
                f"(créé au lancement, hors codec) ; application vérifiée ensuite sur les fichiers, puis par la taille "
                f"décodée (--probe)")
    # --- 3. checkpoint
    ck = Path(R["paths"]["checkpoint_file"])
    if not ck.exists():
        add("checkpoint : fichier", FAIL, f"introuvable : {ck}")
    else:
        add("checkpoint : fichier", OK, f"{ck.name} ({ck.stat().st_size / 1e6:.1f} Mo) ; identifiant {R['ckpt_id']} "
                                         f"(écrit dans l'en-tête du flux)")
        if deep:
            rep = (ckpt_cache or {}).get(R["checkpoint"]) or inspect_checkpoint(ck)
            if ckpt_cache is not None:
                ckpt_cache[R["checkpoint"]] = rep
            if not rep.get("loaded"):
                add("checkpoint : lecture", FAIL, rep.get("error", "non chargé"))
            else:
                add("checkpoint : lecture", OK, f"{rep['python_type']}, clés : {rep.get('top_level_keys', [])[:8]} ; "
                                                f"{rep.get('n_tensors', '?')} tenseurs, {rep.get('n_parameters', '?')} paramètres")
                if rep.get("config_like_fields"):
                    add("checkpoint : champs de configuration trouvés", INFO,
                        json.dumps(dict(list(rep["config_like_fields"].items())[:15]), default=str, ensure_ascii=False))
                else:
                    add("checkpoint : champs de configuration", INFO, "aucun champ scalaire de type configuration "
                                                                      "(rien ne permet de déduire une qualité depuis le fichier)")
        else:
            add("checkpoint : compatibilité avec le modèle", TODO,
                "--deep (inspection des formes) puis chargement réel par le codec (--probe / --run)")
    # --- 6. détecteur
    d = R["detector"]
    if d.get("weights"):
        wp = Path(d["weights"]); wp = wp if wp.is_absolute() else Path(R["paths"]["project_root"]) / wp
        add("détecteur : poids", OK if wp.exists() else FAIL, str(wp))
    add("détecteur : paramètres", INFO, f"hash {R['detector_hash'][:12]} ; {json.dumps(d, ensure_ascii=False, default=str)}")
    # --- énergie / mesure
    E = R["energy"]
    if E["idle_seconds"] < IDLE_MIN_RATIO * E["measure_power_secs"]:
        add("énergie : baseline initiale", FAIL, f"{E['idle_seconds']} s < {IDLE_MIN_RATIO} x measure_power_secs")
    if 0 < E["idle_between_seconds"] < IDLE_MIN_RATIO * E["measure_power_secs"]:
        add("énergie : baseline entre répétitions", FAIL, f"{E['idle_between_seconds']} s < {IDLE_MIN_RATIO} x measure_power_secs")
    if E["tracking_mode"] not in ("machine", "process"):
        add("énergie : tracking_mode", FAIL, E["tracking_mode"])
    add("énergie : méthode", INFO, f"mesure {E['tracking_mode']}, échantillonnage {E['measure_power_secs']} s, canal attendu "
                                   f"{E['expected_channel'] or '(aucun)'}, RAPL exigé : {E['require_confirmed_rapl']} "
                                   f"(confirmation réelle : --check_energy puis démarrage de l'expérience)")
    # --- sortie
    out_dir = Path(R["paths"]["out_dir"])
    if out_dir.exists() and any(out_dir.iterdir()):
        add("dossier de résultats", FAIL, f"{out_dir} existe déjà et n'est pas vide (aucun écrasement)")
    else:
        add("dossier de résultats", OK, str(out_dir))
    # --- durée estimée
    est, f = estimate_params(R)
    per_pass = n_i * (est["enc_I_s"] + est["dec_I_s"]) + (n - n_i) * (est["enc_P_s"] + est["dec_P_s"])
    total = R["repeats"] * per_pass * 1.03 + E["idle_seconds"] + max(0, R["repeats"] - 1) * E["idle_between_seconds"] + \
        E["cooldown_seconds"] + 60
    R["_estimated_seconds"] = total
    add("durée estimée (grossière)", OK if total <= R["limits"]["max_hours_per_config"] * 3600 else FAIL,
        f"{total / 3600:.2f} h (facteur pixels x{f:.2f} ; limite {R['limits']['max_hours_per_config']} h)")
    return out


def cross_checks(Rs, deep=False, ckpt_cache=None):
    """Contrôles entre expériences d'un même lot : détecteur identique, checkpoints compatibles, dossiers distincts."""
    out = []
    dets = {R["id"]: R["detector_hash"] for R in Rs}
    same = len(set(dets.values())) <= 1
    out.append({"check": "détecteur identique entre expériences", "status": OK if same else FAIL,
                "detail": "hash commun " + next(iter(dets.values()))[:12] if same else f"hash différents : { {k: v[:8] for k, v in dets.items()} }"})
    dirs = [R["paths"]["dirname"] for R in Rs]
    out.append({"check": "dossiers de résultats distincts", "status": OK if len(set(dirs)) == len(dirs) else FAIL, "detail": str(dirs)})
    for key in ("energy", "measurement", "codec", "video", "frames_count", "repeats"):
        vals = {R["id"]: canon_hash(R[key]) for R in Rs}
        if len(set(vals.values())) > 1:
            out.append({"check": f"paramètre '{key}' variable entre expériences", "status": WARN,
                        "detail": "ces expériences ne diffèrent pas seulement par GOP/résolution/checkpoint : "
                                  "comparaison à interpréter avec prudence"})
    if deep and ckpt_cache:
        cmp_ = compare_checkpoints(ckpt_cache)
        if cmp_["same_signature"] is None:
            out.append({"check": "checkpoints : compatibilité d'architecture", "status": TODO if len(ckpt_cache) > 1 else INFO,
                        "detail": "signature indisponible ou un seul checkpoint"})
        else:
            # des formes différentes ne prouvent pas une incompatibilité (architectures éventuellement différentes) :
            # le test décisif est le chargement par le codec (--probe / --run) ; ici, simple avertissement de comparabilité
            out.append({"check": "checkpoints : mêmes noms et formes de tenseurs", "status": OK if cmp_["same_signature"] else WARN,
                        "detail": json.dumps(cmp_["signature_groups"], ensure_ascii=False)[:300] +
                                  ("" if cmp_["same_signature"] else " -> checkpoints de structures différentes : "
                                   "comparaison entre eux à justifier ; chargement réel vérifié par --probe")})
        if cmp_["differing_config_fields"]:
            out.append({"check": "checkpoints : champs de configuration différents", "status": INFO,
                        "detail": json.dumps(cmp_["differing_config_fields"], default=str, ensure_ascii=False)[:600]})
    return out


def has_fail(checks):
    return any(c["status"] == FAIL for c in checks)


def print_checks(title, checks):
    sym = {OK: "[ OK ]", FAIL: "[ÉCHEC]", WARN: "[ATTN]", INFO: "[info]", TODO: "[  ?  ]"}
    print(f"\n--- {title}")
    for c in checks:
        print(f"  {sym[c['status']]} {c['check']} : {c['detail']}")


# ------------------------------------------------------------------------------------ espace de noms du moteur
def to_namespace(R, inputs_dir, extra=None):
    est, _ = estimate_params(R)
    E, M, C = R["energy"], R["measurement"], R["codec"]
    a = argparse.Namespace(
        no_frames=R["frames_count"], repeats=R["repeats"], gop=R["gop"],
        checkpoint=R["ckpt_id"], checkpoint_name=R["checkpoint"], checkpoint_file=R["paths"]["checkpoint_file"],
        threads=int(C["threads"]), run_name=R["paths"]["dirname"], start_index=int(R["video"]["start_index"]),
        inp_path=str(inputs_dir), prefix=R["video"]["prefix"], ckpt_dir=R["paths"]["checkpoints_dir"],
        root=R["paths"]["project_root"], results_root=str(Path(R["paths"]["results_root"]) / R["paths"]["results_group"]),
        out_dir=R["paths"]["out_dir"], warmup_frames=int(C["warmup_frames"]),
        idle_seconds=float(E["idle_seconds"]), idle_between_seconds=float(E["idle_between_seconds"]),
        measure_power_secs=float(E["measure_power_secs"]), country=E["country_iso_code"], tracking_mode=E["tracking_mode"],
        cooldown=float(E["cooldown_seconds"]), closed_gop=bool(C["closed_gop"]),
        cov_min=float(M["profiler_coverage_min"]), cov_max=float(M["profiler_coverage_max"]),
        dry_run=False, est_enc_I_s=est["enc_I_s"], est_enc_P_s=est["enc_P_s"], est_dec_I_s=est["dec_I_s"],
        est_dec_P_s=est["dec_P_s"], max_hours=float(R["limits"]["max_hours_per_config"]),
        check_energy=False, check_seconds=20.0, expected_channel=E["expected_channel"] or "",
        allow_unconfirmed_rapl=not bool(E["require_confirmed_rapl"]),
        height=R["H"], width=R["W"], config_id=R["id"], fps=R["fps"], fps_assumed=R["fps_assumed"])
    for k, v in (extra or {}).items():
        setattr(a, k, v)
    return a


def namespace_consistency(R, a):
    """UNE seule configuration par expérience : vérifie que l'espace de noms du moteur `a` reflète EXACTEMENT les
    paramètres résolus du YAML. Retourne la liste des écarts (vide = cohérent)."""
    E, C, M = R["energy"], R["codec"], R["measurement"]
    pairs = [("gop", a.gop, R["gop"]), ("no_frames", a.no_frames, R["frames_count"]), ("repeats", a.repeats, R["repeats"]),
             ("width", a.width, R["W"]), ("height", a.height, R["H"]), ("checkpoint_name", a.checkpoint_name, R["checkpoint"]),
             ("checkpoint(id)", a.checkpoint, R["ckpt_id"]), ("checkpoint_file", a.checkpoint_file, R["paths"]["checkpoint_file"]),
             ("out_dir", a.out_dir, R["paths"]["out_dir"]), ("run_name", a.run_name, R["paths"]["dirname"]),
             ("fps", a.fps, R["fps"]), ("fps_assumed", a.fps_assumed, R["fps_assumed"]),
             ("start_index", a.start_index, int(R["video"]["start_index"])), ("prefix", a.prefix, R["video"]["prefix"]),
             ("closed_gop", a.closed_gop, bool(C["closed_gop"])), ("threads", a.threads, int(C["threads"])),
             ("warmup_frames", a.warmup_frames, int(C["warmup_frames"])),
             ("measure_power_secs", a.measure_power_secs, float(E["measure_power_secs"])),
             ("tracking_mode", a.tracking_mode, E["tracking_mode"]), ("country", a.country, E["country_iso_code"]),
             ("idle_seconds", a.idle_seconds, float(E["idle_seconds"])),
             ("idle_between_seconds", a.idle_between_seconds, float(E["idle_between_seconds"])),
             ("cooldown", a.cooldown, float(E["cooldown_seconds"])), ("expected_channel", a.expected_channel, E["expected_channel"] or ""),
             ("allow_unconfirmed_rapl", a.allow_unconfirmed_rapl, not bool(E["require_confirmed_rapl"])),
             ("cov_min", a.cov_min, float(M["profiler_coverage_min"])), ("cov_max", a.cov_max, float(M["profiler_coverage_max"]))]
    return [f"{n}: moteur={x!r} != YAML={y!r}" for n, x, y in pairs if x != y]
