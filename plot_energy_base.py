
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Résultats des 3 répétitions de l'expérience VCM
repetitions = ["Répétition 1", "Répétition 2", "Répétition 3"]

# Énergie totale en Wh
energie = [27.9656, 30.5719, 29.0335]

# Temps d'exécution en secondes, converti en minutes
temps = [4084.75 / 60, 4784.44 / 60, 5052.86 / 60]

# Énergie par image en mWh/image
energie_image = [192.8661, 210.8405, 200.2308]

# Dossier de sauvegarde
output_dir = Path("energy_results/vcm_base/courbes")
output_dir.mkdir(parents=True, exist_ok=True)


def afficher_courbe(valeurs, titre, ylabel, nom_fichier):
    plt.figure(figsize=(10, 6))

    x = np.arange(len(repetitions))

    barres = plt.bar(x, valeurs, width=0.65)

    plt.xticks(x, repetitions, fontsize=12)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(titre, fontsize=16)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Afficher les valeurs au-dessus des barres
    for barre in barres:
        hauteur = barre.get_height()
        plt.text(
            barre.get_x() + barre.get_width() / 2,
            hauteur,
            f"{hauteur:.2f}",
            ha="center",
            va="bottom",
            fontsize=12
        )

    plt.tight_layout()
    plt.savefig(output_dir / nom_fichier, dpi=300)
    plt.show()


# 1. Comparaison de l'énergie totale
afficher_courbe(
    energie,
    "Comparaison de l'énergie totale du VCM Base Layer",
    "Énergie consommée (Wh)",
    "energie_totale.png"
)

# 2. Comparaison du temps d'exécution
afficher_courbe(
    temps,
    "Comparaison du temps d'exécution du VCM Base Layer",
    "Temps d'exécution (minutes)",
    "temps_execution.png"
)

# 3. Comparaison de l'énergie par image
afficher_courbe(
    energie_image,
    "Comparaison de l'énergie moyenne par image",
    "Énergie (mWh/image)",
    "energie_par_image.png"
)

print("\nLes 3 graphiques ont été enregistrés dans :", output_dir)

