
import numpy as np
import constriction

data = np.load("debug_base4_R_y_block.npz")
probs = np.ascontiguousarray(data["probs"], dtype=np.float32)
symbols = np.ascontiguousarray(data["symbols"], dtype=np.int32)

CHUNK_SIZE = 4096

print("PMF :", probs.shape)
print("Test par blocs de", CHUNK_SIZE, "lignes\n")

for start in range(0, len(symbols), CHUNK_SIZE):
    end = min(start + CHUNK_SIZE, len(symbols))

    p = np.ascontiguousarray(probs[start:end])
    s = np.ascontiguousarray(symbols[start:end])

    print(f"=== Lignes {start} à {end - 1} ===", flush=True)

    try:
        family = constriction.stream.model.Categorical(perfect=False)
        encoder = constriction.stream.queue.RangeEncoder()

        encoder.encode(s, family, p)
        encoded = encoder.get_compressed()

        print("RESULTAT : SUCCES")
        print("Bytes :", len(encoded), "\n", flush=True)

    except BaseException as e:
        print("RESULTAT : ECHEC")
        print(type(e).__name__, str(e), "\n", flush=True)
