"""Entry point of the Assignment 2 starter package: load a community and summarize it.

    python main.py                                   # base case (16 members), day d1
    python main.py --community my_mix                # your own data/params_community_my_mix.json
    python main.py --day d4                          # another day of the week (and that day's cap)
    python main.py --members                         # also print every member's Assignment 1 summary
    python main.py --demo                            # also run the scenario-generation demonstration (Question 3)

No model is provided: build your Assignment 1 model class on the ``InputData`` of each member, then the community
models of the assignment (R, I, C, DP, ...) on top of it, one function per experiment, writing to ``results/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_community, list_communities
from src.reference_model import ReferenceModel
from src.individual_model import IndividualModel
from src.community_model import CommunityModel

RESULTS_DIR = Path(__file__).resolve().parent / "results"

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base", help=f"base (shipped) or your own params_community_<name>.json; available: {list_communities()}")
    ap.add_argument("--day", default="d1", help="d1 (base case) ... d7")
    ap.add_argument("--members", action="store_true", help="print the Assignment 1 summary of every member")
    ap.add_argument("--demo", action="store_true", help="run the scenario-generation demonstration")
    args = ap.parse_args()

    com = load_community(args.community, day=args.day)
    print(com.summary(), "\n")
    if args.members:
        for m in com.members:
            print(m.summary(), "\n")
    print("Cap profile (kW):", " ".join(f"{x:.1f}" for x in com.contract.capacity_limit_kW))
    ic = com.contract.individual_capacity_limit_kW
    print("Individual caps in model I (kW), hour 0 / hour 11:",
          "  ".join(f"{cid} ({com.member(cid).archetype}) {ic[cid][0]:.2f}/{ic[cid][11]:.2f}" for cid in ("C1", "C1_PVB", "C4", "C4_PVB") if cid in com.member_ids))
    (RESULTS_DIR / args.community).mkdir(parents=True, exist_ok=True)
    if args.demo:
        import runpy

    # Reference model
    model_ref = ReferenceModel(com)
    results_ref = model_ref.solve()

    print("\nREFERENCE MODEL (R)")
    print("Daily import :", results_ref["daily_import"])
    print("Peak import  :", results_ref["peak_import"])
    print("Exceedance   :", results_ref["exceedance"])
    print("Total cost   :", results_ref["total_cost"])

    # Individual model
    model_ind = IndividualModel(com)
    results_ind = model_ind.solve()

    print("\nINDIVIDUAL MODEL (I)")
    print("Daily import :", results_ind["daily_import"])
    print("Peak import  :", results_ind["peak_import"])
    print("Exceedance   :", results_ind["exceedance"])
    print("Total cost   :", results_ind["total_cost"])

    # Community model
    model_com = CommunityModel(com)
    results_com = model_com.solve()

    print("\nCOMMUNITY MODEL (C)")
    print("Daily import :", results_com["daily_import"])
    print("Peak import  :", results_com["peak_import"])
    print("Exceedance   :", results_com["exceedance"])
    print("Total cost   :", results_com["total_cost"])
    print("C            :", results_com["C"])
    print("D            :", results_com["D"])
    print("P            :", results_com["P"])
    print("Lambda_t     :", results_com["lambda_t"])
    print(settlement)

if __name__ == "__main__":
    main()
