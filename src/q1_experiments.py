from src.reference_model import ReferenceModel
from src.individual_model import IndividualModel
from src.community_model import CommunityModel

def run_q1(com, verbose=True):

# Reference model
    model_ref = ReferenceModel(com)
    results_ref = model_ref.solve()

    if verbose:
        print("\nREFERENCE MODEL (R)")
        print("Daily import :", results_ref["daily_import"])
        print("Peak import  :", results_ref["peak_import"])
        print("Exceedance   :", results_ref["exceedance"])
        print("Total cost   :", results_ref["total_cost"])

# Individual model
    model_ind = IndividualModel(com)
    results_ind = model_ind.solve()

    if verbose:
        print("\nINDIVIDUAL MODEL (I)")
        print("Daily import :", results_ind["daily_import"])
        print("Peak import  :", results_ind["peak_import"])
        print("Exceedance   :", results_ind["exceedance"])
        print("Total cost   :", results_ind["total_cost"])
        print("Tariff discount ratio:", results_ind["tariff_discount_ratio"])
        print("Penalty rate [DKK/kWh]:", results_ind["penalty_rate"])
        print("Total penalty paid [DKK]:", round(results_ind["total_penalty"], 2))

    penalty_share = (results_ind["total_penalty"] / results_ind["total_cost"])
    
    if verbose:
        print("Penalty share of total cost [%]:", round(100 * penalty_share, 2))

# Community model
    model_com = CommunityModel(com)
    results_com = model_com.solve()

    if verbose:
        print("\nCOMMUNITY MODEL (C)")
        print("Daily import :", results_com["daily_import"])
        print("Peak import  :", results_com["peak_import"])
        print("Exceedance   :", results_com["exceedance"])
        print("Total cost   :", results_com["total_cost"])
        print("Procurment cost :", results_com["C"])
        print("Disutility cost :", results_com["D"])
        print("Penalty cost :", results_com["P"])

    return results_ref, results_ind, results_com