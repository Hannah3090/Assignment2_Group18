import numpy as np
import gurobipy as gp
import pandas as pd
from gurobipy import GRB
import matplotlib.pyplot as plt

from data_loader import load_community
from individual_model import IndividualModel
from community_model import CommunityModel

beta_values = [0.0, 0.25, 0.50, 0.75, 1.00]

results = {
    "beta": [],
    "cost_I": [],
    "cost_C": [],
    "peak_I": [],
    "peak_C": [],
    "exc_I": [],
    "exc_C": []
}

com = load_community("base", day="d1")

for beta in beta_values:

    com.contract.tariff_discount_ratio = beta

    model_I = IndividualModel(com)
    res_I = model_I.solve()

    model_C = CommunityModel(com)
    res_C = model_C.solve()

    results["beta"].append(beta)

    results["cost_I"].append(res_I["total_cost"])

    results["peak_I"].append(res_I["peak_import"])

    results["exc_I"].append(res_I["exceedance"])

    results["cost_C"].append(res_C["total_cost"])

    results["peak_C"].append(res_C["peak_import"])

    results["exc_C"].append(res_C["exceedance"])

#FIGURE IN THE TEXT:

df = pd.DataFrame(results)
print(df)

plt.figure(figsize=(8,4))

plt.plot(df["beta"], df["cost_I"], marker="s", label="Model I")
plt.plot(df["beta"], df["cost_C"], marker="o", label="Model C")

plt.xlabel("Tariff discount ratio (β)")
plt.ylabel("Total cost [DKK]")
plt.ylim(300,600)
plt.title("Community cost under varying tariff discount (β)")

plt.legend()
plt.grid()
plt.show()

# SUBLPOT FOR APPENDIX:

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Peak import:
axes[0].plot(df["beta"], df["peak_I"], marker="s", label="Model I")
axes[0].plot(df["beta"], df["peak_C"], marker="o", label="Model C")

axes[0].set_xlabel("Tariff discount ratio (β)")
axes[0].set_ylabel("Peak import [kW]")
axes[0].set_title("Peak import")
axes[0].grid(True)
axes[0].legend()

# Exceedance:
axes[1].plot(df["beta"], df["exc_I"], marker="s", label="Model I")
axes[1].plot(df["beta"], df["exc_C"], marker="o", label="Model C")

axes[1].set_xlabel("Tariff discount ratio (β)")
axes[1].set_ylabel("Exceedance [kWh]")
axes[1].set_title("Congestion exceedance")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()