src = open("test_iframe_bitstream.py", encoding="utf-8").read()
cut = src.index("# 6. Encodage")

patch = '''
import entropy_models as em
eb = net.if_model.entropy_bottleneck

def _p(input, condition=None, return_sym=False):
    eb._set_condition(condition)
    sym = eb.quantize(input, "symbols", eb.mean)
    cdf, cl, off, idx = eb.get_cdf(input.size())
    rel = (sym - off).long()
    hi = cl.reshape(-1).long()[idx.long().expand(rel.shape)] - 1
    rel = torch.minimum(rel.clamp_min(0), hi)
    sym = (rel + off.long()).to(sym.dtype)
    s = em.ac.range_index_encode(sym - off, cdf, cl, idx)
    return (s, eb.dequantize(sym, eb.mean)) if return_sym else s

eb.compress = _p
print("[diag] compress du bottleneck remplace (h_hat coherent avec le flux)")
'''

exec(compile(src[:cut] + patch + src[cut:], "diag_run", "exec"))