from codecarbon import EmissionsTracker
import time

tracker = EmissionsTracker(
    project_name="VCM_Diagnostic",
    output_dir="./energy_results/diagnostic",
    output_file="diagnostic.csv",
    measure_power_secs=1,
    log_level="info"
)

print("Début du diagnostic...")
tracker.start()

start = time.perf_counter()
for i in range(20_000_000):
    x = i * i
elapsed = time.perf_counter() - start

emissions = tracker.stop()

data = tracker.final_emissions_data

print("\n===== DIAGNOSTIC =====")
print(f"Temps : {elapsed:.2f} s")
print(f"Energie : {data.energy_consumed * 1000:.6f} Wh")
print(f"Emissions : {emissions:.8f} kg CO2")
print("Résultats enregistrés dans energy_results/diagnostic")
