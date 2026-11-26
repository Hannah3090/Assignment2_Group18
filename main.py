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
from src.dynamic_pricing_model import DynamicPricingModel 

from src.q1_experiments import run_q1
from sensitivity_beta import run_beta_sensitivity
from sensitivity_cap import run_cap_sensitivity
from src.plotting_q1 import plot_q1_comparison

from src.q2_experiments import prepare_q2_inputs, compute_q1_member_costs, run_q2_base_case
from src.q2_sensitivity import run_q2_cap_sensitivity
from src.q2_plotting import plot_q2_prices, plot_q2_net_profile, plot_q2_cap_sensitivity


RESULTS_DIR = Path(__file__).resolve().parent / "results"

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--community", default="base", help=f"base (shipped) or your own params_community_<name>.json; available: {list_communities()}")
    ap.add_argument("--day", default="d1", help="d1 (base case) ... d7")
        
    # Question 1
    ap.add_argument("--q1-results", action="store_true")
    ap.add_argument("--q1-plots", action="store_true")
    ap.add_argument("--q1-sensitivity_beta", action="store_true")
    ap.add_argument("--q1-sensitivity_cap", action="store_true")

    # Question 2
    ap.add_argument("--q2-results", action="store_true")
    ap.add_argument("--q2-plots", action="store_true")
    ap.add_argument("--q2-sensitivity", action="store_true")

    args = ap.parse_args()

    com = load_community(args.community, day=args.day)
    print(com.summary(), "\n")
    
    (RESULTS_DIR / args.community).mkdir(parents=True, exist_ok=True)

# Q1 results: 
    if args.q1_results:
        results_ref, results_ind, results_com = run_q1(com, verbose=True)

# Q1 plots: 
    if args.q1_plots:
        results_ref, results_ind, results_com = run_q1(com, verbose=False)
        plot_q1_comparison(results_ref, results_ind, results_com)

 # Q1 sensitivity analysis on beta: 
    if args.q1_sensitivity_beta:
        run_beta_sensitivity()

 # Q1 sensitivity analysis on capacity limit: 
    if args.q1_sensitivity_cap:
        run_cap_sensitivity()

# Q2 results: 
    if args.q2_results:
        members_data, contract_data, day_data = (prepare_q2_inputs(com))
        _, ind_costs_I = compute_q1_member_costs(com)
        q2_results = run_q2_base_case(com, members_data, contract_data, day_data, ind_costs_I, verbose=True)
        # print(q2_results)

# Q2 plots: 
    if args.q2_plots:
        model_com = CommunityModel(com)
        results_com = model_com.solve()
        members_data, contract_data, day_data = (prepare_q2_inputs(com))
        _, ind_costs_I = compute_q1_member_costs(com)
        q2_results = run_q2_base_case(com, members_data, contract_data, day_data, ind_costs_I, verbose=False)

        plot_q2_prices(results_com, q2_results["prices_cap"], q2_results["prices_bb"], q2_results["prices_ir"], q2_results["prices_bb_ir"], day_data)
        plot_q2_net_profile(q2_results["res_cap"], q2_results["res_bb"], q2_results["res_ir"], q2_results["res_bb_ir"], contract_data)

# Q2 sensitivity analysis: 
    if args.q2_sensitivity:
        members_data, contract_data, day_data = (prepare_q2_inputs(com))
        sens_records = run_q2_cap_sensitivity(com, members_data, contract_data, day_data)

        plot_q2_cap_sensitivity(sens_records)

if __name__ == "__main__":
    main()
