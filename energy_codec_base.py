import os
import sys
import csv
import time
import gc
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

INPUT_DIR = ROOT / "videos_5" / "input_png"
CHECKPOINT = ROOT / "checkpoints" / "base" / "checkpoint_base_1.pth.tar"

RESULT_DIR = ROOT / "energy_results" / "vcm_base"
SUMMARY_FILE = RESULT_DIR / "energy_summary.csv"

NO_FRAMES = 145
PREFIX = "frame_"
GOP = 32
REPEATS = 3
CPU_THREADS = 2

RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# VERIFICATIONS
# ============================================================

if not INPUT_DIR.is_dir():
    raise FileNotFoundError(f"Dossier introuvable : {INPUT_DIR}")

if not CHECKPOINT.is_file():
    raise FileNotFoundError(f"Checkpoint introuvable : {CHECKPOINT}")

images = list(INPUT_DIR.glob("*.png"))

if len(images) < NO_FRAMES:
    raise RuntimeError(
        f"Il faut {NO_FRAMES} images, mais seulement {len(images)} sont disponibles."
    )

for i in range(1, NO_FRAMES + 1):
    image_path = INPUT_DIR / f"{PREFIX}{i:03d}.png"
    if not image_path.is_file():
        raise FileNotFoundError(f"Image manquante : {image_path.name}")

# ============================================================
# INITIALISATION
# ============================================================

import torch
from codecarbon import EmissionsTracker

torch.set_num_threads(CPU_THREADS)

print("==============================================")
print(" VCM BASE LAYER - MESURE ENERGETIQUE")
print("==============================================")
print("Images       :", NO_FRAMES)
print("Resolution   : 768 x 432")
print("Checkpoint   :", CHECKPOINT.name)
print("Device       : CPU")
print("Repetitions  :", REPEATS)
print("CodeCarbon   : 3.3.2")
print("==============================================")

import test_base

# ============================================================
# INTERCEPTION DE Pframe.test()
# Le modèle est chargé avant le démarrage du tracker.
# La mesure commence juste avant l'exécution de test().
# ============================================================

ORIGINAL_TEST = test_base.Pframe.test
CURRENT_REPEAT = 0
MEASUREMENTS = []


def read_codecarbon_csv(csv_path):
    if not csv_path.exists():
        return None

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None

    return rows[-1]


def measured_test(self, no_frames, inp_path, prefix, out_path, gop):
    global CURRENT_REPEAT

    repeat_dir = RESULT_DIR / f"repeat_{CURRENT_REPEAT}"
    repeat_dir.mkdir(parents=True, exist_ok=True)

    tracker_file = f"codecarbon_repeat_{CURRENT_REPEAT}.csv"

    tracker = EmissionsTracker(
        project_name=f"VCM_BaseLayer_Repeat_{CURRENT_REPEAT}",
        output_dir=str(RESULT_DIR),
        output_file=tracker_file,
        measure_power_secs=1,
        save_to_file=True
    )

    print(f"\n===== DEBUT MESURE {CURRENT_REPEAT}/{REPEATS} =====")

    result = None
    wall_duration = None

    tracker.start()
    start = time.perf_counter()

    try:
        result = ORIGINAL_TEST(
            self,
            no_frames,
            inp_path,
            prefix,
            out_path,
            gop
        )
        wall_duration = time.perf_counter() - start
        return result

    finally:
        if wall_duration is None:
            wall_duration = time.perf_counter() - start

        tracker.stop()

        csv_path = RESULT_DIR / tracker_file
        last_row = read_codecarbon_csv(csv_path)

        energy_wh = None
        emissions_kg = None

        if last_row:
            if last_row.get("energy_consumed"):
                energy_wh = float(last_row["energy_consumed"]) * 1000

            if last_row.get("emissions"):
                emissions_kg = float(last_row["emissions"])

        row = {
            "repeat": CURRENT_REPEAT,
            "images": no_frames,
            "duration_s": wall_duration,
            "energy_Wh": energy_wh,
            "energy_mWh_per_image": (
                energy_wh * 1000 / no_frames
                if energy_wh is not None else None
            ),
            "emissions_kg_CO2eq": emissions_kg,
            "mse": result[0] if result else None,
            "ssim": result[1] if result else None,
            "bpp": result[2] if result else None,
            "codecarbon_file": tracker_file
        }

        MEASUREMENTS.append(row)

        print(f"\n===== FIN MESURE {CURRENT_REPEAT} =====")
        print("Duree (s) :", wall_duration)
        print("Energie (Wh) :", energy_wh)
        print("Energie (mWh/image) :", row["energy_mWh_per_image"])


# Remplacement temporaire de la méthode pendant cette expérience.
test_base.Pframe.test = measured_test

# ============================================================
# EXECUTION DES REPETITIONS
# ============================================================

for repeat in range(1, REPEATS + 1):
    CURRENT_REPEAT = repeat

    print(f"\n\n######## REPETITION {repeat}/{REPEATS} ########")

    output_dir = RESULT_DIR / f"repeat_{repeat}" / "reconstructed"
    output_dir.mkdir(parents=True, exist_ok=True)

    gc.collect()

    try:
        test_base.encode_the_base_layer(
            base_checkpoint=str(CHECKPOINT),
            no_frames=NO_FRAMES,
            inp_path=str(INPUT_DIR),
            prefix=PREFIX,
            out_path=str(output_dir),
            gop=GOP
        )
    except Exception:
        print(f"\nERREUR pendant la repetition {repeat}.")
        raise

    gc.collect()

# ============================================================
# SAUVEGARDE DU RESUME
# ============================================================

if MEASUREMENTS:
    fields = [
        "repeat",
        "images",
        "duration_s",
        "energy_Wh",
        "energy_mWh_per_image",
        "emissions_kg_CO2eq",
        "mse",
        "ssim",
        "bpp",
        "codecarbon_file"
    ]

    with SUMMARY_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(MEASUREMENTS)

print("\n==============================================")
print("EXPERIENCE TERMINEE")
print("==============================================")
print("Resume :", SUMMARY_FILE)
print("Dossier :", RESULT_DIR)

for row in MEASUREMENTS:
    print(
        f"Run {row['repeat']} | "
        f"Temps={row['duration_s']:.2f} s | "
        f"Energie={row['energy_Wh']} Wh"
    )
