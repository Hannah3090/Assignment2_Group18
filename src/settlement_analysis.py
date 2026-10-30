import numpy as np
import gurobipy as gp
import pandas as pd
from gurobipy import GRB
import matplotlib.pyplot as plt

from data_loader import load_community
from reference_model import ReferenceModel
from individual_model import IndividualModel
from community_model import CommunityModel

com = load_community("base", day="d1")

model_R = ReferenceModel(com)
res_R = model_R.solve(return_members=True)

model_I = IndividualModel(com)
res_I = model_I.solve(return_members=True)

model_C = CommunityModel(com)
res_C = model_C.solve()

lambda_t = res_C["lambda"] # dual analysis for community settlement.

print("\nHourly settlement prices")

for t, lam in enumerate(lambda_t, start=1):
    print(f"Hour {t:02d}: {lam:.4f}")

member_net = res_C["member_net"]

# RESULTS FOR REPORTING:

settlement = {}
cost_R = {}
cost_I = {}

for cid, data in res_R["member_results"].items():

    cost_R[cid] = data["summary"]["objective (2.1) = C + D_L [DKK]"]

for cid, data in res_I["member_results"].items():

    cost_I[cid] = data["summary"]["objective (2.1) = C + D_L [DKK]"]

for cid, net_profile in member_net.items():

    payment = 0

    for t in range(24):

        payment += (lambda_t[t] * net_profile[t])

    settlement[cid] = payment

for cid, value in settlement.items():

    print(cid, round(value,2))

# RESULTS FOR REPORTING:
manager_revenue = sum(settlement.values())
community_cost = res_C["total_cost"]

comparison = pd.DataFrame({"R": cost_R, "I": cost_I, "C": settlement}).T.T

comparison["Gain_vs_R"] = (comparison["R"] - comparison["C"])
comparison["Gain_vs_I"] = (comparison["I"] - comparison["C"])

print(comparison.round(2))
print(manager_revenue)
print(community_cost)
print(manager_revenue - community_cost)

# ----------------------- PLOTTING IN THE SAME FILE--------------------------

#FIGURE IN THE TEXT:
plt.figure(figsize=(12,5))

comparison["Gain_vs_R"].plot(kind="bar", color=["green" if x > 0 else "red" for x in comparison["Gain_vs_R"]])

plt.axhline(y=0, color="black", linewidth=1)
plt.ylabel("Gain [DKK]")
plt.xlabel("Consumers in Community")
plt.title("Individual Gains/Losses with No-Contract-Case")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,5))

comparison["Gain_vs_I"].plot(kind="bar", color=["green" if x > 0 else "red" for x in comparison["Gain_vs_I"]])

plt.axhline(y=0, color="black")
plt.ylabel("Gain [DKK]")
plt.xlabel("Consumers in Community")
plt.title("Individual Gains/Losses with Individual-Contract-Case")
plt.tight_layout()
plt.show()

consumer_type_gain = {"Load only": comparison.loc[["C1","C2","C3","C4"], "Gain_vs_R"].mean(),
                      "PV": comparison.loc[["C1_PV","C2_PV","C3_PV","C4_PV"], "Gain_vs_R"].mean(),
                      "Battery": comparison.loc[["C1_B","C2_B","C3_B","C4_B"],"Gain_vs_R"].mean(),
                      "PV + Battery": comparison.loc[["C1_PVB","C2_PVB","C3_PVB","C4_PVB"],"Gain_vs_R"].mean()}

# PLOTS FOR APPENDIX:

plt.figure(figsize=(8,4))

plt.bar(consumer_type_gain.keys(),consumer_type_gain.values())
plt.axhline(y=0, color="black")
plt.ylabel("Average Gain [DKK]")
plt.title("Average Benefit by Consumer Type")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))

labels = ["Settlement\nRevenue", "Community\nCost"]

values = [manager_revenue, community_cost]

plt.bar(labels, values)
plt.ylabel("DKK")
plt.title("Settlement Revenue vs Community Cost")

plt.tight_layout()
plt.show()