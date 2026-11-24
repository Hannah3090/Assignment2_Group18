import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

Path("figures").mkdir(exist_ok=True)

# Plot 1 (KPI comparison):

models = ["Reference", "Individual", "Community"]

daily_import = [173.386, 167.536, 180.081]
peak_import = [27.60, 11.08, 9.39]
exceedance = [66.35, 4.29, 0.00]
cost = [398.24, 480.67, 372.16]

fig, axs = plt.subplots(2, 2, figsize=(8,6))

bars_imp = axs[0,0].bar(models, daily_import, color="green")

axs[0,0].set_title("Daily Import")
axs[0,0].set_ylabel("kWh")
axs[0,0].set_ylim(0, 200)

for bar in bars_imp:
    height = bar.get_height()
    axs[0,0].text(bar.get_x() + bar.get_width()/2, height + 2, f"{height:.2f}", ha="center", va="bottom")

bars_peak = axs[0,1].bar(models, peak_import, color="orange")

axs[0,1].set_title("Peak Import")
axs[0,1].set_ylabel("kWh")
axs[0,1].set_ylim(0, 200)

for bar in bars_peak:
    height = bar.get_height()
    axs[0,1].text(bar.get_x() + bar.get_width()/2, height + 2, f"{height:.2f}", ha="center", va="bottom")

bars_exc = axs[1,0].bar(models, exceedance, color="red")

axs[1,0].set_title("Congestion Exceedance")
axs[1,0].set_ylabel("kWh")
axs[1,0].set_ylim(0, 200)

for bar in bars_exc:
    height = bar.get_height()

    axs[1,0].text(bar.get_x() + bar.get_width()/2, height + 2, f"{height:.2f}", ha="center", va="bottom")

bars_cost = axs[1,1].bar(models, cost, color="blue")   
axs[1,1].set_ylim(0, 550)

for bar in bars_cost:
    height = bar.get_height()

    axs[1,1].text(bar.get_x() + bar.get_width()/2, height + 2, f"{height:.2f}", ha="center", va="bottom")

axs[1,1].bar(models, cost)
axs[1,1].set_title("Total Cost")
axs[1,1].set_ylabel("DKK")

plt.tight_layout()
plt.savefig("figures/comparison_cases.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot 2 (Community Import-export profile with contract cap):

hours = range(1,25)

community_export = [
    0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,
    5.28,12.0,4.67,
    0,0,0]

community_import = [
    9.39, 9.39, 9.39, 9.39, 9.39, 9.39,
    9.39, 9.39, 9.39, 9.39, 9.39, 9.39,
    9.39, 9.39, 9.39, 9.39, 9.39, 8.64,
    0.0, 0.0, 0.0, 0.0, 2.42, 9.39]

plt.figure(figsize=(10,4))

plt.plot(hours, community_import, marker="o", label="Import")
plt.plot(hours, community_export, marker="s",label="Export")
plt.axhline(y=9.39, color="red",  linestyle="--", label="Contract cap")

plt.xlabel("Hour")
plt.ylabel("kW")
plt.title("Community Import and Export")

plt.legend()
plt.grid(True)
plt.savefig("figures/import_export_profiles_with_cap.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot 3 (Cost decomposition):

labels = ["Procurement", "Disutility", "Penalty"]
values = [324.77, 47.39, 0.0]

plt.figure(figsize=(6,4))
plt.bar(labels, values,color="red")

plt.ylabel("DKK")
plt.title("Cost Components of Community Contract")
plt.savefig("figures/cost_decompositon.png", dpi=300, bbox_inches="tight")
plt.show()
