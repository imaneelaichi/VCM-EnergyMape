import sys
import numpy as np
import constriction

# Charger la ligne exacte qui provoque l'erreur
data = np.load("debug_base4_R_y_block.npz")
probs = data["probs"][30261:30262].copy()
symbols = data["symbols"][30261:30262].astype(np.int32)

mode = sys.argv[1] if len(sys.argv) > 1 else "float64"

if mode == "float64":
    probs = probs.astype(np.float64)

elif mode == "float32_norm":
    probs = probs.astype(np.float32)
    probs = probs / probs.sum(axis=1, keepdims=True)

else:
    raise ValueError("Mode inconnu")

print("Mode :", mode)
print("Shape :", probs.shape)
print("Somme :", probs.sum(axis=1))
print("Symbole :", symbols)
print("Probabilité du symbole :", probs[0, symbols[0]])

family = constriction.stream.model.Categorical(perfect=False)
encoder = constriction.stream.queue.RangeEncoder()

encoder.encode(symbols, family, probs)

print("TEST REUSSI")