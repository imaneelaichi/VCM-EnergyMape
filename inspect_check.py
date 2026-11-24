
import json
import csv
import os

check_file = r"results\VCM_EnergyMap\exp3_v51\_energy_check_v4.json"

if not os.path.exists(check_file):
    print("Fichier introuvable :", check_file)
else:
    with open(check_file, encoding="utf-8") as f:
        data = json.load(f)

    for name in ("idle", "busy"):
        if name not in data:
            continue

        r = data[name]
        print(f"\n--- {name.upper()} ---")

        for key in (
            "wall_s",
            "cc_duration_s",
            "energy_window_s",
            "energy_cpu_j",
            "energy_ram_j",
            "ram_power_w_last",
            "cpu_power_w_last",
        ):
            print(f"{key}: {r.get(key)}")

        ram_power = r.get("ram_power_w_last")
        if ram_power and ram_power > 0:
            print(
                "Fenêtre RAM implicite :",
                r["energy_ram_j"] / ram_power,
                "s"
            )

csv_file = (
    r"results\VCM_EnergyMap\exp3_v4_RAPL_10rep"
    r"\level_1\idle_baselines.csv"
)

if os.path.exists(csv_file):
    print("\n--- BASELINES ---")
    with open(csv_file, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            print(row)
else:
    print("\nCSV des baselines introuvable :", csv_file)
