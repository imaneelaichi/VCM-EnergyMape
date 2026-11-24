
import csv
import json
import re
import statistics
from pathlib import Path

# ============================================================
# 1. CHEMINS
# ============================================================

directory = Path(
    "results/VCM_EnergyMap/exp3_v3_final_10rep/level_1"
)

metadata_path = directory / "metadata.json"
input_path = directory / "energy_tasks.csv"
output_path = directory / "energy_tasks_with_net.csv"
summary_path = directory / "energy_net_summary.csv"

# ============================================================
# 2. CHARGEMENT
# ============================================================

with metadata_path.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

with input_path.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

baselines = {
    item["name"]: item
    for item in metadata["idle_baselines"]
}

print("Nombre de lignes chargées :", len(rows))

# ============================================================
# 3. VERIFICATION
# ============================================================

counts = {}

for row in rows:
    match = re.match(r"^L1_r(\d+)_", row["task"])
    if match:
        repeat = int(match.group(1))
        counts[repeat] = counts.get(repeat, 0) + 1

if len(rows) != 210 or len(counts) != 10:
    raise ValueError("Les données ne correspondent pas aux 10 répétitions.")

if any(counts.get(i, 0) != 20 for i in range(10)):
    raise ValueError(f"Nombre de tâches incorrect : {counts}")

# ============================================================
# 4. CALCUL ENERGIE NETTE
# ============================================================

for row in rows:

    match = re.match(r"^L1_r(\d+)_", row["task"])

    if not match:
        continue

    repeat = int(match.group(1))

    baseline_name = (
        "idle_initial"
        if repeat == 0
        else f"idle_before_rep{repeat:02d}"
    )

    if baseline_name not in baselines:
        raise ValueError(f"Baseline absente : {baseline_name}")

    baseline = baselines[baseline_name]

    duration = float(row["cc_duration_s"])

    gross_total = float(row["energy_total_j"])
    gross_cpu = float(row["energy_cpu_j"])
    gross_ram = float(row["energy_ram_j"])

    idle_total = float(baseline["power_total_w"])
    idle_cpu = float(baseline["power_cpu_w"])
    idle_ram = float(baseline["power_ram_w"])

    row["repeat"] = repeat
    row["baseline_used"] = baseline_name

    row["idle_power_total_w"] = idle_total

    row["energy_idle_j"] = idle_total * duration
    row["energy_net_j"] = gross_total - row["energy_idle_j"]

    row["energy_idle_cpu_j"] = idle_cpu * duration
    row["energy_net_cpu_j"] = gross_cpu - row["energy_idle_cpu_j"]

    row["energy_idle_ram_j"] = idle_ram * duration
    row["energy_net_ram_j"] = gross_ram - row["energy_idle_ram_j"]

# ============================================================
# 5. SAUVEGARDE DES MESURES INDIVIDUELLES
# ============================================================

fieldnames = list(dict.fromkeys(
    key for row in rows for key in row.keys()
))

with output_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\nCSV individuel :", output_path)

# ============================================================
# 6. REGROUPEMENT PAR CATEGORIE
# ============================================================

groups = {}

for row in rows:

    if "repeat" not in row:
        continue

    match = re.search(r"_(enc|dec)_frame(\d+)$", row["task"])

    if not match:
        continue

    operation = match.group(1).upper()
    frame_number = int(match.group(2))
    frame_type = "I" if frame_number == 0 else "P"

    category = f"{operation}_{frame_type}"

    groups.setdefault(category, []).append(row)

# ============================================================
# 7. STATISTIQUES
# ============================================================

def mean_std(values):
    return (
        statistics.mean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0
    )

summary = []

metrics = [
    ("Temps (s)", "cc_duration_s"),
    ("Energie brute totale (J)", "energy_total_j"),
    ("Energie nette totale (J)", "energy_net_j"),
    ("Energie brute CPU (J)", "energy_cpu_j"),
    ("Energie nette CPU (J)", "energy_net_cpu_j"),
    ("Energie brute RAM (J)", "energy_ram_j"),
    ("Energie nette RAM (J)", "energy_net_ram_j"),
]

print("\n" + "=" * 90)
print("RESULTATS STATISTIQUES : ENERGIE BRUTE ET NETTE")
print("=" * 90)

for category in ["ENC_I", "ENC_P", "DEC_I", "DEC_P"]:

    data = groups.get(category, [])

    if not data:
        continue

    print(f"\n--- {category} (n={len(data)}) ---")

    result = {
        "categorie": category,
        "n": len(data)
    }

    for label, key in metrics:

        values = [float(row[key]) for row in data]
        mean, std = mean_std(values)

        result[label + "_moyenne"] = mean
        result[label + "_ecart_type"] = std

        print(
            f"{label:28s} : "
            f"{mean:12.3f} ± {std:10.3f}"
        )

    summary.append(result)

# ============================================================
# 8. EXPORT DU TABLEAU RECAPITULATIF
# ============================================================

summary_fields = list(summary[0].keys())

with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary)

print("\n" + "=" * 90)
print("VERIFICATION FINALE")
print("=" * 90)

print("Répétitions :", len(counts))
print("Tâches :", sum(counts.values()))
print("Lignes :", len(rows))
print("CSV détaillé :", output_path.resolve())
print("CSV récapitulatif :", summary_path.resolve())
print("\nCalcul terminé.")
