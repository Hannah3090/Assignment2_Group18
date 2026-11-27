import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_community
from src.reference_model import ReferenceModel
from src.individual_model import IndividualModel
from src.community_model import CommunityModel

def run_cap_sensitivity():

    cap_factors = [0.6, 0.8, 1.0, 1.2, 1.4]

    results = {
        "cap_factor": [],

        "delta_cost_I": [],
        "delta_cost_C": [],

        "delta_peak_I": [],
        "delta_peak_C": [],

        "delta_exc_I": [],
        "delta_exc_C": []
    }

    base_com = load_community("base", day="d1")

    reference_model = ReferenceModel(base_com)
    res_R = reference_model.solve()

    cost_R = res_R["total_cost"]
    peak_R = res_R["peak_import"]
    exc_R = res_R["exceedance"]

    base_cap = base_com.contract.capacity_limit_kW.copy()

    for factor in cap_factors:

        com = load_community("base", day="d1")

        com.contract.capacity_limit_kW = (
            factor * base_cap
        )

        for cid in com.contract.individual_capacity_limit_kW:

            com.contract.individual_capacity_limit_kW[cid] = (factor * com.contract.individual_capacity_limit_kW[cid])

        print(
            f"factor = {factor:.1f}, "
            f"cap range = "
            f"{com.contract.capacity_limit_kW.min():.2f} - "
            f"{com.contract.capacity_limit_kW.max():.2f}"
        )

        model_I = IndividualModel(com)
        res_I = model_I.solve()

        model_C = CommunityModel(com)
        res_C = model_C.solve()

        results["cap_factor"].append(factor)

        results["delta_cost_I"].append(res_I["total_cost"] - cost_R)

        results["delta_cost_C"].append(res_C["total_cost"] - cost_R)

        results["delta_peak_I"].append(res_I["peak_import"] - peak_R)

        results["delta_peak_C"].append(res_C["peak_import"] - peak_R)

        results["delta_exc_I"].append(res_I["exceedance"] - exc_R)

        results["delta_exc_C"].append(res_C["exceedance"] - exc_R)

    df = pd.DataFrame(results)

    print("\n")
    print(df)

    #-------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(df["cap_factor"], df["delta_cost_I"], marker="s", label="Model I")

    axes[0].plot(df["cap_factor"], df["delta_cost_C"], marker="o", label="Model C")

    axes[0].axhline(0, color="black", linestyle="--")

    axes[0].set_xlabel("Cap factor")

    axes[0].set_ylabel("Δ Cost relative to R [DKK]")

    axes[0].set_title("Community perspective")

    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df["cap_factor"], df["delta_peak_I"], marker="s", label="Model I")

    axes[1].plot(df["cap_factor"], df["delta_peak_C"], marker="o", label="Model C")

    axes[1].axhline(0, color="black", linestyle="--")

    axes[1].set_xlabel("Cap factor")

    axes[1].set_ylabel("Δ Peak Import relative to R [kW]")

    axes[1].set_title("DSO perspective")

    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    #-------------------------------------------------------------------------------------
    plt.figure(figsize=(6,4))

    plt.plot(df["cap_factor"], df["delta_exc_I"], marker="s", label="Model I")

    plt.plot(df["cap_factor"], df["delta_exc_C"], marker="o", label="Model C")

    plt.axhline(0, color="black", linestyle="--")

    plt.xlabel("Cap factor")

    plt.ylabel("Δ Exceedance relative to R [kWh]")

    plt.title("Congestion Reduction")

    plt.legend()
    plt.grid(True)
    plt.savefig( "figures/sensitivity_cap.png", dpi=300, bbox_inches="tight")
    print(f"\n[Success] Saved all figures to figures/.")
    plt.close()


    return df
