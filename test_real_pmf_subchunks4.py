
import numpy as np
import constriction

data = np.load("debug_base4_R_y_block.npz")
probs = np.ascontiguousarray(data["probs"], dtype=np.float32)
symbols = np.ascontiguousarray(data["symbols"], dtype=np.int32)

ranges = [
    (30208, 30464),
    (30464, 30720),
]

for start, end in ranges:
    p = np.ascontiguousarray(probs[start:end])
    s = np.ascontiguousarray(symbols[start:end])

    row_sums = p.sum(axis=1)
    min_idx = int(np.argmin(row_sums))

    print(f"\n=== Lignes {start} à {end - 1} ===", flush=True)
    print("Somme minimale :", float(row_sums[min_idx]))
    print("Ligne minimale :", start + min_idx)
    print("Nombre de zéros :", int(np.sum(p == 0)))
    print("Minimum positif :", float(p[p > 0].min()))
    print("Nombre de valeurs subnormales :", int(np.sum((p > 0) & (p < np.finfo(np.float32).tiny))))

    try:
        family = constriction.stream.model.Categorical(perfect=False)
        encoder = constriction.stream.queue.RangeEncoder()
        encoder.encode(s, family, p)

        print("RESULTAT : SUCCES")
        print("Bytes :", len(encoder.get_compressed()), flush=True)

    except BaseException as e:
        print("RESULTAT : ECHEC")
        print(type(e).__name__, str(e), flush=True)
