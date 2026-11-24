
import numpy as np
import constriction
from importlib.metadata import version

print("Constriction version:", version("constriction"))

# TEST 1 : une distribution individuelle
p = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

model = constriction.stream.model.Categorical(p, perfect=False)
print("\nTEST 1 : Categorical individuel : OK")

# TEST 2 : création de la famille de modèles
family = constriction.stream.model.Categorical(perfect=False)
print("TEST 2 : Categorical model family : OK")

# TEST 3 : encodage avec une matrice de probabilités
probs = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.4, 0.3, 0.2, 0.1],
    [0.25, 0.25, 0.25, 0.25],
], dtype=np.float32)

symbols = np.array([3, 0, 2], dtype=np.int32)

encoder = constriction.stream.queue.RangeEncoder()
encoder.encode(symbols, family, probs)

encoded = encoder.get_compressed()

print("TEST 3 : RangeEncoder avec matrice : OK")
print("Encoded bytes:", len(encoded))

print("\nTOUS LES TESTS TERMINES")
