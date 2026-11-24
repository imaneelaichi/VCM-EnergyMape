p = "entropy_models.py"
s = open(p, encoding="utf-8", newline="").read()
eol = "\r\n" if "\r\n" in s else "\n"

old = """        strings = ac.range_index_encode(symbols - offset, cdf, cdf_length, idx)

        if return_sym:
            return strings, self.dequantize(symbols, self.mean)""".replace("\n", eol)

new = """        # Correction A : clamp explicite, meme borne que ac_compat (hi = cdf_length - 1)
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

        if return_sym:
            return strings, self.dequantize(symbols, self.mean)""".replace("\n", eol)

assert s.count(old) == 1, "bloc introuvable ou non unique : rien modifie"
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new))
print("OK : entropy_models.py modifie")