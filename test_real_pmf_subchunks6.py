
import numpy as np
import constriction

data = np.load("debug_base4_R_y_block.npz")
probs = np.ascontiguousarray(data["probs"], dtype=np.float32)
symbols = np.ascontiguousarray(data["symbols"], dtype=np.int32)

ranges = [
    (30208, 30272),
    (30272, 30336),
]

for start, end in ranges:
    p = np.ascontiguousarray(probs[start:end])
    s = np.ascontiguousarray(symbols[start:end])

    row_sums = p.sum(axis=1)
    selected = p[np.arange(len(s)), s]

    print(f"\n=== Lignes {start} à {end - 1} ===", flush=True)
    print("Somme minimale :", float(row_sums.min()))
    print("Nombre de zéros :", int(np.sum(p == 0)))
    print("Valeurs subnormales :", int(np.sum(
        (p > 0) & (p < np.finfo(np.float32).tiny)
    )))
    print("Probabilité minimale sélectionnée :", float(selected.min()))
    print("Probabilité maximale sélectionnée :", float(selected.max()))

    try:
        family = constriction.stream.model.Categorical(perfect=False)
        encoder = constriction.stream.queue.RangeEncoder()
        encoder.encode(s, family, p)

        print("RESULTAT : SUCCES")
        print("Bytes :", len(encoder.get_compressed()), flush=True)

    except BaseException as e:
        print("RESULTAT : ECHEC")
        print(type(e).__name__, str(e), flush=True)
