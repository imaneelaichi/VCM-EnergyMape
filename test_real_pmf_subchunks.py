
import numpy as np
import constriction

data = np.load("debug_base4_R_y_block.npz")
probs = np.ascontiguousarray(data["probs"], dtype=np.float32)
symbols = np.ascontiguousarray(data["symbols"], dtype=np.int32)

ranges = [
    (28672, 30720),
    (30720, 32768),
]

for start, end in ranges:
    p = np.ascontiguousarray(probs[start:end])
    s = np.ascontiguousarray(symbols[start:end])

    print(f"\n=== Lignes {start} à {end - 1} ===", flush=True)
    print("Shape :", p.shape)
    print("Somme minimale :", p.sum(axis=1).min())
    print("Minimum positif :", p[p > 0].min())
    print("Nombre de zéros :", np.sum(p == 0))

    try:
        family = constriction.stream.model.Categorical(perfect=False)
        encoder = constriction.stream.queue.RangeEncoder()
        encoder.encode(s, family, p)
        encoded = encoder.get_compressed()

        print("RESULTAT : SUCCES")
        print("Bytes :", len(encoded), flush=True)

    except BaseException as e:
        print("RESULTAT : ECHEC")
        print(type(e).__name__, str(e), flush=True)
