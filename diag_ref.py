src = open("test_iframe_bitstream.py", encoding="utf-8").read()
cut = src.index("# 6. Encodage")
tail = '''
import torch
with torch.no_grad():
    rec_hat, _, _ = net.if_model.compress(aligned, return_hat=True)
rec_hat = align.resume(rec_hat).clamp(0, 1)
mse = torch.mean((original - rec_hat) ** 2).item()
import math
print("[ref] MSE =", mse, "| PSNR =", 10 * math.log10(1 / mse), "dB")
'''
exec(compile(src[:cut] + tail, "diag_run", "exec"))