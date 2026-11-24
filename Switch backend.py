r"""
switch_backend.py -- bascule entropy_models.py / context_model.py entre deux etats (jamais melanges) :

  --to constriction : etat ORIGINAL du depot (plus de 'import ac_compat', plus de clamp)  -> aucun torchac,
                      aucun compilateur Visual Studio necessaire ; requis pour le pipeline constriction (patch 9 fichiers).
  --to ac_compat    : remet vos modifications (import ac_compat + clamp du patch A).

  python switch_backend.py "<...>\\svc_code" --to constriction            # SIMULATION
  python switch_backend.py "<...>\\svc_code" --to constriction --apply    # applique (copies dans local_modifications\\)
Chaque fichier est copie avant modification dans <svc_code>\\local_modifications\\<AAAAMMJJ_HHMMSS>\\. Les fins de ligne
(CRLF/LF) sont preservees. Si un bloc attendu est introuvable, RIEN n'est modifie.
"""
import argparse, shutil, sys, time
from pathlib import Path

NEW_BLOCK = '''        # Correction A : clamp explicite, meme borne que ac_compat (hi = cdf_length - 1)
        sym_rel = symbols - offset
        lens = cdf_length.reshape(-1).long()[idx.long().expand(sym_rel.shape)]
        hi = (lens - 1).to(sym_rel.dtype)
        sym_clamped = torch.minimum(sym_rel.clamp_min(0), hi)
        self.last_n_clamped = int((sym_clamped != sym_rel).sum())
        self.last_n_symbols = sym_rel.numel()
        if self.last_n_clamped:
            import warnings
            warnings.warn(f"bottleneck: {self.last_n_clamped}/{self.last_n_symbols} symboles clampes")
        symbols = sym_clamped + offset

        strings = ac.range_index_encode(sym_clamped, cdf, cdf_length, idx)
'''
OLD_BLOCK = '''        strings = ac.range_index_encode(symbols - offset, cdf, cdf_length, idx)
'''
EDITS = {  # fichier -> liste (texte_original_depot, texte_ac_compat)
 "entropy_models.py": [("#from torchac.torchac import ac\n", "import ac_compat as ac\n"), (OLD_BLOCK, NEW_BLOCK)],
 "context_model.py": [("import torch.nn.functional as F\n", "import torch.nn.functional as F\nimport ac_compat as ac\n")],
}

def plan(path, to):
    raw = path.read_bytes().decode("utf-8"); eol = "\r\n" if "\r\n" in raw else "\n"
    s = raw.replace("\r\n", "\n")
    for orig, comp in EDITS[path.name]:
        old, new = (comp, orig) if to == "constriction" else (orig, comp)
        if path.name == "context_model.py" and to == "constriction":
            old, new = "import ac_compat as ac\n", ""
        if s.count(old) != 1:
            if s.count(new) >= 1 and (to != "constriction" or path.name != "context_model.py" or "import ac_compat" not in s):
                continue  # deja dans l'etat voulu
            return None, f"bloc attendu introuvable (ou non unique) dans {path.name}: {old[:50]!r}"
        s = s.replace(old, new)
    return s.replace("\n", eol), None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("svc_code"); ap.add_argument("--to", choices=["constriction", "ac_compat"], required=True)
    ap.add_argument("--apply", action="store_true"); a = ap.parse_args(); d = Path(a.svc_code).resolve()
    out = {}
    for name in EDITS:
        new, err = plan(d / name, a.to)
        if err: print("ABANDON, rien modifie :", err); sys.exit(1)
        out[name] = new
    for name, new in out.items():
        changed = new.encode("utf-8") != (d / name).read_bytes()
        print(f"{name}: {'sera modifie' if changed else 'deja dans l etat voulu'}")
    if not a.apply: print("SIMULATION : rien modifie. Ajouter --apply."); return
    stamp = time.strftime("%Y%m%d_%H%M%S"); bk = d / "local_modifications" / stamp; n = 1
    while bk.exists(): n += 1; bk = d / "local_modifications" / f"{stamp}_{n}"
    bk.mkdir(parents=True)
    for name, new in out.items():
        shutil.copy2(d / name, bk / name); (d / name).write_bytes(new.encode("utf-8"))
    print(f"OK. Copies avant modification : {bk}\nTest : python -c \"import entropy_models, context_model; print('import OK')\"")
if __name__ == "__main__": main()