import ac_compat, torch

_orig = ac_compat.range_index_encode

def _wrap(symbols, cdf, cdf_length, idx):
    sym = symbols.cpu().long()
    ii = idx.cpu().long().expand(sym.shape)
    lens = cdf_length.cpu().reshape(-1).long()[ii]
    bad = (sym < 0) | (sym > lens - 1)
    print("\n[diag] symboles:", tuple(sym.shape), "| hors plage:", int(bad.sum()))
    for p in bad.nonzero():
        t = tuple(p.tolist())
        print("[diag] pos", t, "| symbole decale =", int(sym[t]),
              "| plage autorisee = [0,", int(lens[t]) - 1, "]")
    return _orig(symbols, cdf, cdf_length, idx)

ac_compat.range_index_encode = _wrap

src = open("test_iframe_bitstream.py", encoding="utf-8").read()
head = src[:src.index("# Calculer la taille")]
exec(compile(head, "diag_run", "exec"))