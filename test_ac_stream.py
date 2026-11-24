import torch
import ac_compat as ac

torch.manual_seed(0)
C, S, N = 8, 21, 500
cdfs, syms = [], []
for _ in range(N):
    p = torch.rand(C, S) + 0.001
    p = (p / p.sum(1, keepdim=True) * 65536).round().clamp_min(1)
    c = torch.cumsum(p, 1)
    c[:, -1] = 65536
    a = torch.cat([c[:, :1], c], 1)                 # même structure que get_cdf
    cdfs.append(a.t().t().to(torch.int32).to(torch.int16))
    syms.append(torch.randint(0, S, (C,)).short())

for c, s in zip(cdfs, syms):
    ac.range_encode(s, c)
bs = ac.flush()
ac.set_string(bs)
ok = all(torch.equal(ac.range_decode(c).short(), s) for c, s in zip(cdfs, syms))
print(len(bs), "octets pour", N * C, "symboles ->", ok)