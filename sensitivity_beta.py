import numpy as np
import gurobipy as gp
import pandas as pd
from gurobipy import GRB
import matplotlib.pyplot as plt

from src.data_loader import load_community
from src.individual_model import IndividualModel
from src.community_model import CommunityModel
from src.reference_model import ReferenceModel

#beta_values = [0.0, 0.25, 0.50, 0.75, 1.00]

beta_values = np.arange(0.0, 1.01, 0.05)

#cap_factor = [0.6,0.8,1.0,1.2,1.4]

results = {
    "beta": [],
    "delta_cost_I": [],
    "delta_cost_C": [],
    "delta_peak_I": [],
    "delta_peak_C": [],
    "delta_exc_I": [],
    "delta_exc_C": []
}

com = load_community("base", day="d1")

reference_model = ReferenceModel(com)
res_R = reference_model.solve()

cost_R = res_R["total_cost"]
peak_R = res_R["peak_import"]
exc_R = res_R["exceedance"]

for beta in beta_values:

    com.contract.tariff_discount_ratio = beta

    model_I = IndividualModel(com)
    res_I = model_I.solve()

    model_C = CommunityModel(com)
    res_C = model_C.solve()

    results["beta"].append(beta)

    results["delta_cost_I"].append(res_I["total_cost"]-cost_R)

    results["delta_peak_I"].append(res_I["peak_import"]-peak_R)

    results["delta_exc_I"].append(res_I["exceedance"]-exc_R)

    results["delta_cost_C"].append(res_C["total_cost"]-cost_R)

    results["delta_peak_C"].append(res_C["peak_import"]-peak_R)

    results["delta_exc_C"].append(res_C["exceedance"]-exc_R)


#FIGURE IN THE TEXT:

df = pd.DataFrame(results)
print(df)

df["sign"] = np.sign(df["delta_cost_C"])

for i in range(len(df)-1):
    y1 = df["delta_cost_C"][i]
    y2 = df["delta_cost_C"][i+1]

    if y1 * y2 < 0:

        beta1 = df["beta"][i]
        beta2 = df["beta"][i+1]

        beta_star = beta1 + (0-y1)*(beta2-beta1)/(y2-y1)

        print(f"Estimated break-even β* = {beta_star:.3f}")

plt.figure(figsize=(8,4))

plt.plot(df["beta"], df["delta_cost_I"], marker="s", label="Model I")
plt.plot(df["beta"], df["delta_cost_C"], marker="o", label="Model C")

plt.xlabel("Tariff discount ratio (β)")
plt.ylabel("Total cost [DKK]")
plt.axhline(0,color="black",linestyle="--")
#plt.ylim(300,600)
plt.title("Community cost under varying tariff discount (β)")
plt.title("Δ Community Cost Relative to Reference Model")

plt.legend()
plt.grid()
plt.show()

# SUBLPOT FOR APPENDIX:

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Peak import:
axes[0].plot(df["beta"], df["delta_peak_I"], marker="s", label="Model I")
axes[0].plot(df["beta"], df["delta_peak_C"], marker="o", label="Model C")

axes[0].set_xlabel("Tariff discount ratio (β)")
axes[0].set_ylabel("Peak import [kW]")
axes[0].set_title("Peak import")
axes[0].axhline(0,color="black",linestyle="--")
axes[0].grid(True)
axes[0].legend()

# Exceedance:
axes[1].plot(df["beta"], df["delta_exc_I"], marker="s", label="Model I")
axes[1].plot(df["beta"], df["delta_exc_C"], marker="o", label="Model C")

axes[1].set_xlabel("Tariff discount ratio (β)")
axes[1].set_ylabel("Exceedance [kWh]")
axes[1].set_title("Congestion exceedance")
axes[1].axhline(0,color="black",linestyle="--")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()
