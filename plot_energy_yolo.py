from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Dossier contenant les resultats
BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "energy_results" / "yolo_145"

CSV_FILE = RESULT_DIR / "energy_summary.csv"

# Charger les resultats
df = pd.read_csv(CSV_FILE)

# Verifier les donnees
required_columns = [
    "mode",
    "duration_s",
    "energy_Wh",
    "energy_mWh_per_image",
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Colonne manquante : {column}")

# Ordre des categories
df["mode"] = df["mode"].map({
    "original": "Images originales",
    "reconstructed": "Images reconstruites"
})

if df["mode"].isna().any():
    raise ValueError("Mode inconnu dans le CSV.")

df = df.set_index("mode").loc[
    ["Images originales", "Images reconstruites"]
].reset_index()

# Fonction pour creer un graphique
def create_bar_chart(column, ylabel, title, filename):
    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        df["mode"],
        df[column],
        width=0.55
    )

    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars:
        value = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    output = RESULT_DIR / filename
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Graphique cree : {output}")

# Graphique 1 : energie totale
create_bar_chart(
    "energy_Wh",
    "Energie (Wh)",
    "Consommation energetique totale de YOLOv5s",
    "energie_totale.png"
)

# Graphique 2 : energie par image
create_bar_chart(
    "energy_mWh_per_image",
    "Energie (mWh/image)",
    "Energie moyenne par image",
    "energie_par_image.png"
)

# Graphique 3 : temps de detection
create_bar_chart(
    "duration_s",
    "Temps (secondes)",
    "Temps total de detection YOLOv5s",
    "temps_detection.png"
)

print("\nVisualisation terminee.")
