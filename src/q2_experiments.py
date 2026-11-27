import gurobipy as gp
from gurobipy import GRB

from src.dynamic_pricing_model import DynamicPricingModel

def compute_q1_member_costs(data, cap_scale=1.0):
    """Computes per-member costs under Reference (R) and Individual Contract (I) at a given cap_scale."""
    c = data.contract
    ref_costs, ind_costs = [], []
    for m in data.members:
        cap_i = [float(v) * float(cap_scale) for v in c.individual_capacity_limit_kW[m.consumer_id]]
        for mode in ("R", "I"):
            rm = gp.Model()
            rm.setParam('OutputFlag', 0)
            l_r = rm.addVars(24, lb=m.load_lower_kWh, ub=m.load_upper_kWh)
            d_r = rm.addVars(24, lb=0.0)
            p_imp_r = rm.addVars(24, lb=0.0)
            p_exp_r = rm.addVars(24, lb=0.0)
            e_exc_r = rm.addVars(24, lb=0.0)
            g_r = rm.addVars(24, lb=0.0, ub=m.pv_available) if m.has_pv else {t: 0.0 for t in range(24)}
            if m.has_battery:
                pch_r = rm.addVars(24, lb=0.0, ub=m.battery_max_charge_kW)
                pdis_r = rm.addVars(24, lb=0.0, ub=m.battery_max_discharge_kW)
                s_r = rm.addVars(24, lb=0.0, ub=m.battery_capacity_kWh)
            else:
                pch_r = pdis_r = s_r = {t: 0.0 for t in range(24)}

            min_e = float(m.min_daily_energy_kWh) if m.min_daily_energy_kWh is not None else float(sum(m.reference_load))
            rm.addConstr(gp.quicksum(l_r[t] for t in range(24)) >= min_e)
            beta_eff = float(c.tariff_discount_ratio) if mode == "I" else 0.0
            alpha_eff = float(c.penalty_DKK_per_kWh) if mode == "I" else 0.0

            for t in range(24):
                rm.addConstr(l_r[t] - float(m.reference_load[t]) <= d_r[t])
                rm.addConstr(float(m.reference_load[t]) - l_r[t] <= d_r[t])
                rm.addConstr(p_imp_r[t] - p_exp_r[t] == l_r[t] + pch_r[t] - pdis_r[t] - g_r[t])
                if mode == "I":
                    rm.addConstr(e_exc_r[t] >= p_imp_r[t] - cap_i[t])
                if m.has_battery:
                    prev_s = float(m.battery_initial_soc_kWh) if t == 0 else s_r[t - 1]
                    rm.addConstr(
                        s_r[t] == prev_s
                        + float(m.battery_charging_efficiency) * pch_r[t]
                        - (1.0 / float(m.battery_discharging_efficiency)) * pdis_r[t]
                    )
                    if t == 23:
                        rm.addConstr(s_r[t] == float(m.battery_initial_soc_kWh))

            rm.setObjective(
                gp.quicksum(
                    float(data.energy_price[t]) * (p_imp_r[t] - p_exp_r[t])
                    + (1.0 - beta_eff) * float(data.import_tariff) * p_imp_r[t]
                    + (1.0 - beta_eff) * float(data.export_tariff) * p_exp_r[t]
                    + alpha_eff * e_exc_r[t]
                    + float(m.linear_disutility) * d_r[t]
                    + (float(m.pv_marginal_cost) * g_r[t] if m.has_pv else 0.0)
                    for t in range(24)
                ),
                GRB.MINIMIZE,
            )
            rm.optimize()
            if mode == "R":
                ref_costs.append(float(rm.ObjVal))
            else:
                ind_costs.append(float(rm.ObjVal))
    return ref_costs, ind_costs


def solve_centralized_scaled(members_data, contract_data, day_data, cap_scale=1.0):
    """Solves Centralized Community Model (C) at a given cap_scale for sensitivity comparisons."""
    m = gp.Model("Centralized_Scaled")
    m.setParam('OutputFlag', 0)
    H = range(24)
    n_m = len(members_data)
    cap = [float(v) * float(cap_scale) for v in contract_data['cap']]

    P_imp = m.addVars(H, lb=0.0)
    P_exp = m.addVars(H, lb=0.0)
    E_exc = m.addVars(H, lb=0.0)

    q, l, delta, p_ch, p_dis, s, g = {}, {}, {}, {}, {}, {}, {}
    for idx, i in enumerate(members_data):
        m.addConstr(gp.quicksum(l[idx, t] for t in H) >= i['min_daily_energy']) if False else None
        for t in H:
            q[idx, t] = m.addVar(lb=-GRB.INFINITY)
            l[idx, t] = m.addVar(lb=i['l_min'][t], ub=i['l_max'][t])
            delta[idx, t] = m.addVar(lb=0.0)
            g[idx, t] = m.addVar(lb=0.0, ub=i['pv_max'][t]) if i['has_pv'] else 0.0
            if i['has_battery']:
                p_ch[idx, t] = m.addVar(lb=0.0, ub=i['p_ch_max'])
                p_dis[idx, t] = m.addVar(lb=0.0, ub=i['p_dis_max'])
                s[idx, t] = m.addVar(lb=i['s_min'], ub=i['s_max'])
            else:
                p_ch[idx, t] = p_dis[idx, t] = s[idx, t] = 0.0

    for idx, i in enumerate(members_data):
        m.addConstr(gp.quicksum(l[idx, t] for t in H) >= i['min_daily_energy'])
        for t in H:
            m.addConstr(q[idx, t] == l[idx, t] + p_ch[idx, t] - p_dis[idx, t] - g[idx, t])
            m.addConstr(l[idx, t] - i['l_ref'][t] <= delta[idx, t])
            m.addConstr(i['l_ref'][t] - l[idx, t] <= delta[idx, t])
            if i['has_battery']:
                prev_s = i['s_init'] if t == 0 else s[idx, t - 1]
                m.addConstr(s[idx, t] == prev_s + i['eta_ch'] * p_ch[idx, t] - (1.0 / i['eta_dis']) * p_dis[idx, t])
                if t == 23:
                    m.addConstr(s[idx, t] == i['s_init'])

    for t in H:
        m.addConstr(gp.quicksum(q[idx, t] for idx in range(n_m)) == P_imp[t] - P_exp[t])
        m.addConstr(E_exc[t] >= P_imp[t] - cap[t])

    m.setObjective(
        gp.quicksum(
            day_data['p_da'][t] * (P_imp[t] - P_exp[t])
            + (1.0 - contract_data['beta']) * day_data['tau_imp'] * P_imp[t]
            + (1.0 - contract_data['beta']) * day_data['tau_exp'] * P_exp[t]
            + contract_data['alpha'] * E_exc[t]
            for t in H
        ) + gp.quicksum(
            members_data[idx]['c_L'] * delta[idx, t] + (members_data[idx]['c_pv'] * g[idx, t] if members_data[idx]['has_pv'] else 0.0)
            for idx in range(n_m) for t in H
        ),
        GRB.MINIMIZE,
    )
    m.optimize()
    return float(m.ObjVal)

def prepare_q2_inputs(com):

    """
    Convert CommunityData object into the dictionaries
    required by DynamicPricingModel.
    """

    c = com.contract

    day_data = {'p_da': [float(x) for x in com.energy_price],
                'tau_imp': float(com.import_tariff),
                'tau_exp': float(com.export_tariff),}

    contract_data = {'beta': float(c.tariff_discount_ratio),
                     'alpha': float(c.penalty_DKK_per_kWh),
                     'cap': [float(x) for x in c.capacity_limit_kW],}

    members_data = []

    for m in com.members:

        l_ref = [float(x) for x in m.reference_load]

        l_min = ([float(x) for x in m.load_lower_kWh]
                if m.load_lower_kWh is not None
                else [float(m.load_min_kWh)] * 24)

        l_max = ([float(x) for x in m.load_upper_kWh]
                 if m.load_upper_kWh is not None
                else [float(m.load_max_kWh)] * 24)

        min_daily_e = (float(m.min_daily_energy_kWh)
                       if m.min_daily_energy_kWh is not None
                       else sum(l_ref))

        member_dict = {'id': m.consumer_id,
                       'l_ref': l_ref,
                       'min_daily_energy': min_daily_e,
                       'c_L': float(m.linear_disutility),
                       'l_min': l_min,
                       'l_max': l_max,
                       'has_pv': m.has_pv,
                       'pv_max': ([float(x) for x in m.pv_available]
                                   if m.has_pv
                                   else [0.0] * 24),
                       'c_pv': (float(m.pv_marginal_cost)
                                if m.has_pv
                                else 0.0),
                        'has_battery': m.has_battery,}

        if m.has_battery:

            member_dict.update({'p_ch_max': float(m.battery_max_charge_kW),
                                'p_dis_max': float(m.battery_max_discharge_kW),
                                 's_min': 0.0,
                                 's_max': float(m.battery_capacity_kWh),
                                 's_init': float(m.battery_initial_soc_kWh),
                                 'eta_ch': float(m.battery_charging_efficiency),
                                 'eta_dis': float(m.battery_discharging_efficiency),})

        members_data.append(member_dict)

    return members_data, contract_data, day_data

def run_q2_base_case(com, members_data, contract_data, day_data, ind_costs_I, verbose=True):

    if verbose:
        print("\n[Step 1/5] Solving DP_cap (Cap only)...")
    dp_cap = DynamicPricingModel(members_data, contract_data, day_data, enable_bb=False, enable_ir=False)
    dp_cap.build_model()
    prices_cap, _ = dp_cap.solve()
    res_cap = dp_cap.extract_summary_metrics()
    
    if verbose:
        print("[Step 2/5] Solving DP_BB (Budget Balance)...")
    dp_bb = DynamicPricingModel(members_data, contract_data, day_data, enable_bb=True, enable_ir=False)
    dp_bb.build_model()
    prices_bb, _ = dp_bb.solve()
    res_bb = dp_bb.extract_summary_metrics()

    if verbose:
        print("[Step 3/5] Solving DP_IR (Individual Rationality vs Model I)...")
    dp_ir = DynamicPricingModel(members_data, contract_data, day_data, enable_bb=False, enable_ir=True)
    dp_ir.build_model(ir_bounds=ind_costs_I)
    prices_ir, _ = dp_ir.solve()
    res_ir = dp_ir.extract_summary_metrics()

    if verbose:
        print("[Step 4/5] Solving DP_BB+IR (Combined BB + IR under uniform prices)...")
    dp_bb_ir = DynamicPricingModel(members_data, contract_data, day_data, enable_bb=True, enable_ir=True)
    dp_bb_ir.build_model(ir_bounds=ind_costs_I)
    prices_bb_ir, _ = dp_bb_ir.solve()
    res_bb_ir = dp_bb_ir.extract_summary_metrics()

    blocking_base = []
    if res_bb_ir is None:
        print("   -> Uniform DP_BB+IR infeasible! Identifying blocking members via minimum IR slack...")
        dp_relax = DynamicPricingModel(members_data, contract_data, day_data, enable_bb=True, enable_ir=True, relax_ir_slack=True)
        dp_relax.build_model(ir_bounds=ind_costs_I)
        dp_relax.solve()
        res_relax = dp_relax.extract_summary_metrics()
        blocking_base = res_relax['blocking_members'] if res_relax else []

    if verbose:
        print("[Step 5/5] Solving DP_BB+IR_disc (Per-consumer price discrimination x_{i,t})...")
    dp_disc = DynamicPricingModel(members_data, contract_data, day_data, enable_bb=True, enable_ir=True, allow_discrimination=True)
    dp_disc.build_model(ir_bounds=ind_costs_I)
    _, _ = dp_disc.solve()
    res_disc = dp_disc.extract_summary_metrics()

    if verbose:
        print("\nDP_cap total cost :", res_cap["total_cost"])
        print("DP_BB total cost  :", res_bb["total_cost"])
        print("DP_IR total cost  :", res_ir["total_cost"])

    if res_bb_ir:
        print("DP_BB+IR total cost :", res_bb_ir["total_cost"])
    else:
        print("DP_BB+IR infeasible")

    print("DP_disc total cost :", res_disc["total_cost"])   

    return {
    "prices_cap": prices_cap,
    "prices_bb": prices_bb,
    "prices_ir": prices_ir,
    "prices_bb_ir": prices_bb_ir,

    "res_cap": res_cap,
    "res_bb": res_bb,
    "res_ir": res_ir,
    "res_bb_ir": res_bb_ir,
    "res_disc": res_disc,

    "blocking_base": blocking_base,
    "dp_ir_model": dp_ir}



