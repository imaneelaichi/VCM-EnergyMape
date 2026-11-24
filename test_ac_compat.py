import torch
import ac_compat as ac

C, L = 4, 20
lengths = torch.tensor([20, 15, 12, 18])
pmf = torch.zeros(C, L + 1)
for c in range(C):
    n = int(lengths[c])
    p = torch.rand(n) + 0.01
    pmf[c, :n] = 0.999 * p / p.sum()
    pmf[c, n] = 0.001                      # masse de queue, comme dans votre code

cdf = ac.pmf2cdf(pmf)
cdf_length = lengths.to(ac.dtype)
shape = (1, C, 32, 32)
idx = torch.arange(C).view(1, C, 1, 1).expand(shape).to(ac.dtype)
symbols = torch.floor(torch.rand(shape) * lengths.view(1, C, 1, 1)).to(ac.dtype)

bs = ac.range_index_encode(symbols, cdf, cdf_length, idx)
rec = ac.range_index_decode(bs, cdf, cdf_length, idx)
print(len(bs), "octets pour", symbols.numel(), "symboles ->", torch.equal(rec, symbols))