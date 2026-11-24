r"""
cleanup_archive.py -- range (ne SUPPRIME PAS) les fichiers de test/doublons dans un dossier d'archive.

  python cleanup_archive.py "<dossier_projet>"                 # SIMULATION : liste seulement, ne touche a rien
  python cleanup_archive.py "<dossier_projet>" --apply         # deplace vers <dossier_projet>\_archive_AAAAMMJJ\
  python cleanup_archive.py "<dossier_projet>" --restore "<dossier_projet>\_archive_AAAAMMJJ"   # remet tout en place

<dossier_projet> = le dossier qui CONTIENT svc_code (ex. ...\sauvegarde_VCM\sauvegarde_VCM).
Securites : (1) simulation par defaut ; (2) refuse d'archiver un module encore importe par un script restant ;
(3) un MANIFEST.csv (chemin, raison, SHA256) permet de tout restaurer ; (4) rien n'est ecrase.
NON touches volontairement : ac_compat.py (importe par entropy_models.py/context_model.py : voir switch_backend.py),
export.py (importe par models/common.py de YOLO), input/ + labels/ + out/ (2e sequence de 200 images), svc_env*.
"""
import argparse, csv, hashlib, re, shutil, sys, time
from pathlib import Path

ITEMS = [  # chemin relatif au dossier projet, raison
 ("svc_code/ac_compat.py.bak_20261112", "copie identique de ac_compat.py"),
 ("svc_code/entropy_models.py.bak_20261112", "doublon exact de entropy_models.py.bak_avant_correction (celui-ci est conserve)"),
 ("svc_code/test_base_backup.py", "sauvegarde de test_base.py"),
 ("svc_code/patch_A.py", "script one-shot deja applique (clamp)"),
 ("svc_code/diag_consistent.py", "diagnostic du chemin ac_compat"),
 ("svc_code/diag_oor.py", "diagnostic du chemin ac_compat"),
 ("svc_code/diag_ref.py", "diagnostic du chemin ac_compat"),
 ("svc_code/diag_strings.py", "diagnostic du chemin ac_compat"),
 ("svc_code/test_ac_compat.py", "test synthetique torchac"),
 ("svc_code/test_ac_stream.py", "test synthetique constriction/contexte"),
 ("svc_code/energy_diagnostic.py", "test CodeCarbon (boucle x*x)"),
 ("svc_code/test_energy.py", "test CodeCarbon (boucle x*x)"),
 ("svc_code/dsd", "doublon exact de README.md"),
 ("svc_code/Audit inventory · PY", "copie de audit_inventory.py"),
 ("svc_code/compare_detection.py", "version 5 images, remplacee par compare_detection_145.py"),
 ("svc_code/videos_5/results/detection_comparison", "test sur 5 images"),
 ("svc_code/videos_5/results/detection_comparison_backup_20261110", "doublon exact de detection_comparison"),
 ("svc_code/energy_results/emissions.csv", "test CodeCarbon d'une AUTRE machine (i7-8550U)"),
 ("energy_results/diagnostic", "test CodeCarbon (boucle x*x)"),
]
SKIP_DIRS = {"svc_env", "svc_env_hp", ".venv", "_archive", "__pycache__", ".git"}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def files_of(p):
    return [p] if p.is_file() else [q for q in p.rglob("*") if q.is_file()]

def remaining_imports(root, archived_py):
    bad = []
    for f in (root / "svc_code").rglob("*.py"):
        if any(part.startswith(("svc_env", "_archive", ".venv")) for part in f.relative_to(root).parts): continue
        if f in archived_py: continue
        for m in re.finditer(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", f.read_text(encoding="utf-8", errors="ignore"), re.M):
            for a in archived_py:
                if m.group(1) == a.stem: bad.append((str(f.relative_to(root)), a.name))
    return bad

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--apply", action="store_true"); ap.add_argument("--restore")
    a = ap.parse_args(); root = Path(a.root).resolve()
    if a.restore:
        arc = Path(a.restore).resolve()
        for r in csv.DictReader(open(arc / "MANIFEST.csv", encoding="utf-8-sig")):
            src, dst = arc / r["path"], root / r["path"]
            if not src.exists(): print("absent de l'archive :", r["path"]); continue
            if dst.exists(): print("REFUSE (existe deja) :", r["path"]); continue
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(src), str(dst)); print("restaure :", r["path"])
        return
    plan = [(Path(r), why) for r, why in ITEMS if (root / r).exists()]
    absent = [r for r, _ in ITEMS if not (root / r).exists()]
    archived_py = {root / r for r, _ in plan if str(r).endswith(".py")}
    bad = remaining_imports(root, archived_py)
    if bad:
        print("ABANDON : ces scripts restants importent un module a archiver :"); [print("  ", b) for b in bad]; sys.exit(1)
    print(f"{'APPLICATION' if a.apply else 'SIMULATION'} : {len(plan)} elements a archiver ({len(absent)} deja absents)")
    for r, why in plan: print(f"  {r.as_posix()}   <- {why}")
    if not a.apply:
        print("\nRien n'a ete deplace. Relancer avec --apply pour archiver."); return
    arc = root / f"_archive_{time.strftime('%Y%m%d')}"; arc.mkdir(exist_ok=True)
    man = []
    for r, why in plan:
        for f in files_of(root / r):
            man.append({"path": str(f.relative_to(root)).replace("\\", "/"), "reason": why, "size": f.stat().st_size, "sha256": sha(f)})
    with open(arc / "MANIFEST.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "reason", "size", "sha256"]); w.writeheader(); w.writerows(man)
    for r, _ in plan:
        dst = arc / r
        if dst.exists(): print("REFUSE (existe deja dans l'archive) :", r); continue
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(root / r), str(dst))
    print(f"\nOK -> {arc}  (MANIFEST.csv). Pour annuler : --restore \"{arc}\"")
if __name__ == "__main__": main()