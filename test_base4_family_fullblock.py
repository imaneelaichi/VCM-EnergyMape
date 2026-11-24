
import sys
import numpy as np
import constriction

data = np.load("debug_base4_R_y_block.npz")
probs = data["probs"].copy()
symbols = data["symbols"].astype(np.int32)

mode = sys.argv[1] if len(sys.argv) > 1 else "float64"

if mode == "float64":
    probs = probs.astype(np.float64)

elif mode == "float32_norm":
    probs = probs.astype(np.float32)
    row_sums = probs.sum(axis=1, keepdims=True)

    if not np.all(np.isfinite(row_sums)) or np.any(row_sums <= 0):
        raise ValueError("Somme de PMF invalide")

    probs = probs / row_sums

else:
    raise ValueError("Mode inconnu")

print("Mode :", mode)
print("Shape :", probs.shape)
print("Symboles :", symbols.shape)
print("Sommes min/max :", probs.sum(axis=1).min(),
      probs.sum(axis=1).max())
print("Probabilités finies :", np.isfinite(probs).all())
print("Probabilités négatives :", np.sum(probs < 0))

family = constriction.stream.model.Categorical(perfect=False)
encoder = constriction.stream.queue.RangeEncoder()

encoder.encode(symbols, family, probs)

print("TEST COMPLET REUSSI")
