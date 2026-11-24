
from codecarbon import EmissionsTracker
import time

# Initialisation du suivi énergétique
tracker = EmissionsTracker(
    project_name="Energy_Measurement",
    output_dir="./energy_results",
    measure_power_secs=1,
    log_level="error"
)

print("Début de la mesure énergétique...")

# Démarrage
tracker.start()

start_time = time.time()

# ==================================================
# PLACE TON PROGRAMME À MESURER ICI
# Exemple :
for i in range(10000000):
    x = i * i
# ==================================================

execution_time = time.time() - start_time

# Arrêt de la mesure
emissions = tracker.stop()

# Récupération des résultats
energy_kwh = tracker.final_emissions_data.energy_consumed
energy_wh = energy_kwh * 1000

print("\n========== RESULTATS ==========")
print(f"Temps d'exécution : {execution_time:.2f} secondes")
print(f"Energie consommée : {energy_wh:.6f} Wh")
print(f"Energie consommée : {energy_kwh:.9f} kWh")
print(f"Emissions CO2 : {emissions:.8f} kg")
print("================================")
