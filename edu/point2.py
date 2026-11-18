import os
import sys
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# Dynamic search for the directory containing 'src' and 'data'
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = None

search_dir = current_file_dir
while search_dir and search_dir != os.path.dirname(search_dir):
    for root, dirs, files in os.walk(search_dir):
        if "src" in dirs and "data" in dirs:
            project_root = root
            break
    if project_root:
        break
    search_dir = os.path.dirname(search_dir)

if project_root and project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import data_loader safely
try:
    from src.data_loader import load_community
except ImportError as e:
    print(f"[Error] Could not find 'src' directory in project path hierarchy: {current_file_dir}")
    print(f"Details: {e}")
    sys.exit(1)


class DynamicPricingModel:
    """
    Formulation of Question 2: Dynamic Pricing (DP) as a single-level MILP.
    Replaces lower-level member LPs with their KKT optimality conditions 
    and Big-M linearization.
    """
    def __init__(self, community_data, contract_data, day_data, 
                 enable_bb=False, enable_ir=False, M=100.0, cap_override=None, alpha_override=None):
        self.members = community_data
        self.contract = contract_data.copy()
        if cap_override is not None:
            self.contract['cap'] = [cap_override] * 24
        if alpha_override is not None:
            self.contract['alpha'] = alpha_override
            
        self.day = day_data
        self.enable_bb = enable_bb
        self.enable_ir = enable_ir
        self.M = M
        
        self.hours = list(range(24))
        self.model = gp.Model("DynamicPricing_DP")
        
        # Solver parameters
        self.model.setParam('OutputFlag', 0)    # Suppress verbose log for batch sensitivity
        self.model.setParam('TimeLimit', 45)   # Max 45s per solve
        self.model.setParam('MIPGap', 0.01)    # 1% MIP gap
        
    def build_model(self, ref_costs_I=None):
        m = self.model
        H = self.hours
        I = self.members
        
        # 1. Upper-level decision variables (Community Manager)
        x = m.addVars(H, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x_price")
        P_imp = m.addVars(H, lb=0.0, name="P_imp")
        P_exp = m.addVars(H, lb=0.0, name="P_exp")
        E_exc = m.addVars(H, lb=0.0, name="E_exc")
        
        # 2. Lower-level primal variables
        q = {}
        l = {}
        delta = {}
        p_ch = {}
        p_dis = {}
        s = {}
        g = {}
        
        for i_idx, i in enumerate(I):
            for t in H:
                q[i_idx, t] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"q_{i_idx}_{t}")
                l[i_idx, t] = m.addVar(lb=i.get('l_min', 0.0), ub=i.get('l_max', 3.0), name=f"l_{i_idx}_{t}")
                delta[i_idx, t] = m.addVar(lb=0.0, ub=1.0, name=f"delta_{i_idx}_{t}")
                
                if i.get('has_battery', False):
                    p_ch_max = i.get('p_ch_max', 0.75)
                    p_dis_max = i.get('p_dis_max', 0.75)
                    s_min = i.get('s_min', 0.2)
                    s_max = i.get('s_max', 2.0)
                    
                    p_ch[i_idx, t] = m.addVar(lb=0.0, ub=p_ch_max, name=f"p_ch_{i_idx}_{t}")
                    p_dis[i_idx, t] = m.addVar(lb=0.0, ub=p_dis_max, name=f"p_dis_{i_idx}_{t}")
                    s[i_idx, t] = m.addVar(lb=s_min, ub=s_max, name=f"s_{i_idx}_{t}")
                else:
                    p_ch[i_idx, t], p_dis[i_idx, t], s[i_idx, t] = 0.0, 0.0, 0.0
                    
                if i.get('has_pv', False) and 'pv_max' in i:
                    pv_val = i['pv_max']
                    pv_limit = pv_val[t] if isinstance(pv_val, list) else pv_val
                    g[i_idx, t] = m.addVar(lb=0.0, ub=pv_limit, name=f"g_{i_idx}_{t}")
                else:
                    g[i_idx, t] = 0.0

        # 3. KKT Dual & Binary variables for Big-M linearization
        mu_dev_pos = {}
        mu_dev_neg = {}
        z_dev_pos = {}
        z_dev_neg = {}
        mu_energy = {}
        z_energy = {}
        
        for i_idx, i in enumerate(I):
            mu_energy[i_idx] = m.addVar(lb=0.0, name=f"mu_energy_{i_idx}")
            z_energy[i_idx] = m.addVar(vtype=GRB.BINARY, name=f"z_energy_{i_idx}")
            
            for t in H:
                mu_dev_pos[i_idx, t] = m.addVar(lb=0.0, name=f"mu_dev_pos_{i_idx}_{t}")
                mu_dev_neg[i_idx, t] = m.addVar(lb=0.0, name=f"mu_dev_neg_{i_idx}_{t}")
                z_dev_pos[i_idx, t] = m.addVar(vtype=GRB.BINARY, name=f"z_dev_pos_{i_idx}_{t}")
                z_dev_neg[i_idx, t] = m.addVar(vtype=GRB.BINARY, name=f"z_dev_neg_{i_idx}_{t}")

        # 4. Upper-level grid balance and capacity constraints
        for t in H:
            m.addConstr(gp.quicksum(q[i_idx, t] for i_idx in range(len(I))) == P_imp[t] - P_exp[t], name=f"grid_bal_{t}")
            m.addConstr(E_exc[t] >= P_imp[t] - self.contract['cap'][t], name=f"cap_exc_{t}")

        # 5. Lower-level KKT conditions per member
        for i_idx, i in enumerate(I):
            E_ref_total = sum(i['l_ref'])
            m.addConstr(gp.quicksum(l[i_idx, t] for t in H) >= E_ref_total, name=f"energy_min_{i_idx}")
            
            # KKT Complementary Slackness for daily minimum energy requirement
            m.addConstr(gp.quicksum(l[i_idx, t] for t in H) - E_ref_total <= self.M * z_energy[i_idx])
            m.addConstr(mu_energy[i_idx] <= self.M * (1 - z_energy[i_idx]))
            
            for t in H:
                m.addConstr(q[i_idx, t] == l[i_idx, t] + p_ch[i_idx, t] - p_dis[i_idx, t] - g[i_idx, t])
                
                if i.get('has_battery', False):
                    eta_ch = i.get('eta_ch', 0.95)
                    eta_dis = i.get('eta_dis', 0.95)
                    s_max = i.get('s_max', 2.0)
                    s_init = i.get('s_init', 0.5 * s_max)
                    
                    if t == 0:
                        m.addConstr(s[i_idx, t] == s_init + eta_ch * p_ch[i_idx, t] - (1 / eta_dis) * p_dis[i_idx, t])
                    else:
                        m.addConstr(s[i_idx, t] == s[i_idx, t-1] + eta_ch * p_ch[i_idx, t] - (1 / eta_dis) * p_dis[i_idx, t])

                # Stationarity condition: dL/dl = x_t + mu_pos - mu_neg - mu_energy = 0
                m.addConstr(x[t] + mu_dev_pos[i_idx, t] - mu_dev_neg[i_idx, t] - mu_energy[i_idx] == 0)
                m.addConstr(mu_dev_pos[i_idx, t] + mu_dev_neg[i_idx, t] <= i['c_L'])

                # Complementary slackness: positive deviation constraint
                m.addConstr(l[i_idx, t] - i['l_ref'][t] - delta[i_idx, t] <= 0)
                m.addConstr(delta[i_idx, t] - l[i_idx, t] + i['l_ref'][t] <= self.M * z_dev_pos[i_idx, t])
                m.addConstr(mu_dev_pos[i_idx, t] <= self.M * (1 - z_dev_pos[i_idx, t]))

                # Complementary slackness: negative deviation constraint
                m.addConstr(i['l_ref'][t] - l[i_idx, t] - delta[i_idx, t] <= 0)
                m.addConstr(delta[i_idx, t] - i['l_ref'][t] + l[i_idx, t] <= self.M * z_dev_neg[i_idx, t])
                m.addConstr(mu_dev_neg[i_idx, t] <= self.M * (1 - z_dev_neg[i_idx, t]))

        # 6. Budget Balance constraint (DP_BB variant)
        if self.enable_bb:
            for t in H:
                dso_cost_t = (
                    self.day['p_da'][t] * (P_imp[t] - P_exp[t]) +
                    (1 - self.contract['beta']) * self.day['tau_imp'] * P_imp[t] +
                    (1 - self.contract['beta']) * self.day['tau_exp'] * P_exp[t] +
                    self.contract['alpha'] * E_exc[t]
                )
                internal_revenue_t = gp.quicksum(
                    mu_energy[i_idx] * (sum(I[i_idx]['l_ref']) / 24.0) +
                    mu_dev_pos[i_idx, t] * I[i_idx]['l_ref'][t] - 
                    I[i_idx]['c_L'] * delta[i_idx, t]
                    for i_idx in range(len(I))
                )
                m.addConstr(internal_revenue_t == dso_cost_t, name="BudgetBalance")

        # 7. Individual Rationality constraint (DP_IR variant)
        if self.enable_ir and ref_costs_I is not None:
            for i_idx, i in enumerate(I):
                member_cost_i = gp.quicksum(x[t] * q[i_idx, t] + i['c_L'] * delta[i_idx, t] for t in H)
                m.addConstr(member_cost_i <= ref_costs_I[i_idx], name=f"IR_member_{i_idx}")

        # 8. Social Welfare Minimization Objective Function
        total_dso_cost = gp.quicksum(
            self.day['p_da'][t] * (P_imp[t] - P_exp[t]) +
            (1 - self.contract['beta']) * self.day['tau_imp'] * P_imp[t] +
            (1 - self.contract['beta']) * self.day['tau_exp'] * P_exp[t] +
            self.contract['alpha'] * E_exc[t]
            for t in H
        )
        total_disutility = gp.quicksum(I[i_idx]['c_L'] * delta[i_idx, t] for i_idx in range(len(I)) for t in H)
        m.setObjective(total_dso_cost + total_disutility, GRB.MINIMIZE)

    def solve(self):
        self.model.optimize()
        if self.model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            posted_prices = [self.model.getVarByName(f"x_price[{t}]").X for t in self.hours]
            return posted_prices, self.model.ObjVal
        else:
            return None, None

    def extract_member_metrics(self, ref_costs_I):
        """Extract per-member detailed metrics (Table 12 requirement)."""
        metrics = []
        for i_idx, i in enumerate(self.members):
            m_cost = sum(
                self.model.getVarByName(f"x_price[{t}]").X * self.model.getVarByName(f"q_{i_idx}_{t}").X +
                i['c_L'] * self.model.getVarByName(f"delta_{i_idx}_{t}").X
                for t in self.hours
            )
            disutil = sum(i['c_L'] * self.model.getVarByName(f"delta_{i_idx}_{t}").X for t in self.hours)
            baseline = ref_costs_I[i_idx]
            savings = baseline - m_cost
            
            metrics.append({
                'member_id': i['id'],
                'actual_cost': m_cost,
                'disutility': disutil,
                'ref_cost': baseline,
                'savings': savings
            })
        return metrics

    def detect_indifferent_hours(self):
        """Locate member-hours where stationarity allows flat/indifferent response."""
        indifferent = []
        for i_idx, i in enumerate(self.members):
            c_L = i['c_L']
            for t in self.hours:
                mu_pos = self.model.getVarByName(f"mu_dev_pos_{i_idx}_{t}").X
                mu_neg = self.model.getVarByName(f"mu_dev_neg_{i_idx}_{t}").X
                # When duals do not hit strict bounds, response set is flat
                if abs(mu_pos - c_L) < 1e-3 or abs(mu_neg - c_L) < 1e-3:
                    indifferent.append((i_idx, t))
        return indifferent

def extract_net_import(self):
        """Extract the 24-hour net community power import (P_imp[t] - P_exp[t])."""
        if self.model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return None
        return [
            self.model.getVarByName(f"P_imp[{t}]").X - self.model.getVarByName(f"P_exp[{t}]").X
            for t in self.hours
        ]


def main():
    print("=== Loading Community Base Case Data ===")
    data = load_community("base", day="d1")
    
    if isinstance(data, tuple):
        community, contract = data
    else:
        community = getattr(data, 'members', data)
        contract = getattr(data, 'dso_contract', getattr(data, 'contract', None))

    first_member = community[0]
    
    day_data = {
        'p_da': getattr(first_member, 'day_ahead_price_dkk_kwh', getattr(first_member, 'day_ahead_price', [0.5]*24)),
        'tau_imp': getattr(first_member, 'grid_tariff_imp_dkk_kwh', getattr(first_member, 'grid_tariff_imp', 0.5)),
        'tau_exp': getattr(first_member, 'grid_tariff_exp_dkk_kwh', getattr(first_member, 'grid_tariff_exp', 0.4)),
    }
    
    contract_data = {
        'beta': getattr(contract, 'tariff_discount_ratio', 0.75),
        'alpha': getattr(contract, 'penalty_DKK_per_kWh', 10.0),
        'cap': getattr(contract, 'capacity_limit_kw', getattr(contract, 'capacity_limit', [9.4]*24)),
    }

    members_data = []
    ref_costs_I = []
    
    for member in community:
        l_ref = getattr(member, 'reference_load', getattr(member, 'reference_load_kwh', None))
        
        m_dict = {
            'id': getattr(member, 'consumer_id', f"member_{len(members_data)}"),
            'l_ref': l_ref,
            'c_L': getattr(member, 'linear_disutility', 0.5),
            'l_min': getattr(member, 'load_lower_kwh', getattr(member, 'load_lower', 0.0)),
            'l_max': getattr(member, 'load_upper_kwh', getattr(member, 'load_upper', 3.0)),
            'has_battery': getattr(member, 'has_battery', False),
            'has_pv': getattr(member, 'has_pv', False),
        }
        
        if m_dict['has_battery'] and hasattr(member, 'battery') and member.battery is not None:
            bat = member.battery
            m_dict.update({
                'p_ch_max': getattr(bat, 'p_ch_max', getattr(bat, 'p_max', 0.75)),
                'p_dis_max': getattr(bat, 'p_dis_max', getattr(bat, 'p_max', 0.75)),
                's_min': getattr(bat, 's_min', 0.2),
                's_max': getattr(bat, 's_max', 2.0),
                's_init': getattr(bat, 's_init', 1.0),
                'eta_ch': getattr(bat, 'eta_ch', 0.95),
                'eta_dis': getattr(bat, 'eta_dis', 0.95),
            })
            
        if m_dict['has_pv'] and hasattr(member, 'pv') and member.pv is not None:
            pv_gen = getattr(member.pv, 'available_generation', getattr(member.pv, 'available_generation_kwh', [0.0]*24))
            m_dict['pv_max'] = pv_gen
            
        members_data.append(m_dict)
        
        # Calculate reference benchmark cost per member under standard retail pricing
        ref_cost_i = sum((day_data['p_da'][t] + day_data['tau_imp']) * l_ref[t] for t in range(24))
        ref_costs_I.append(ref_cost_i)

    # 1. Solve Base Case Variants
    print("\n[1/3] Solving DP_cap (Cap only)...")
    dp_cap = DynamicPricingModel(members_data, contract_data, day_data, enable_bb=False, enable_ir=False)
    dp_cap.build_model()
    prices_cap, obj_cap = dp_cap.solve()

    print("[2/3] Solving DP_BB (Budget Balance)...")
    dp_bb = DynamicPricingModel(members_data, contract_data, day_data, enable_bb=True, enable_ir=False)
    dp_bb.build_model()
    prices_bb, obj_bb = dp_bb.solve()

    print("[3/3] Solving DP_IR (Individual Rationality)...")
    dp_ir = DynamicPricingModel(members_data, contract_data, day_data, enable_bb=False, enable_ir=True)
    dp_ir.build_model(ref_costs_I=ref_costs_I)
    prices_ir, obj_ir = dp_ir.solve()

    # --- Summary Table ---
    print("\n" + "="*55)
    print(f"{'Dynamic Pricing Variant':<25} | {'Objective Value (DKK)':<25}")
    print("="*55)
    print(f"{'DP_cap (Cap only)':<25} | {obj_cap:.2f} DKK" if obj_cap is not None else f"{'DP_cap':<25} | Infeasible")
    print(f"{'DP_BB (Budget Balance)':<25} | {obj_bb:.2f} DKK" if obj_bb is not None else f"{'DP_BB':<25} | Infeasible")
    print(f"{'DP_IR (Ind. Rationality)':<25} | {obj_ir:.2f} DKK" if obj_ir is not None else f"{'DP_IR':<25} | Infeasible")
    print("="*55)

    # --- TABLE 12 METRICS: PER-MEMBER BREAKDOWN ---
    if obj_ir is not None:
        print("\n=== Per-Member Metrics & Savings under DP_IR (Table 12 Requirement) ===")
        print(f"{'Member ID':<12} | {'Actual Cost (DKK)':<18} | {'Disutility (DKK)':<16} | {'Ref Cost (DKK)':<15} | {'Savings (DKK)':<15}")
        print("-" * 84)
        ir_metrics = dp_ir.extract_member_metrics(ref_costs_I)
        blocking_members = []
        for m in ir_metrics:
            is_binding = abs(m['actual_cost'] - m['ref_cost']) < 1e-2
            if is_binding:
                blocking_members.append(m['member_id'])
            print(f"{m['member_id']:<12} | {m['actual_cost']:<18.2f} | {m['disutility']:<16.2f} | {m['ref_cost']:<15.2f} | {m['savings']:<15.2f}")
        
        print("\n[Analysis] Blocking Members (Active IR Constraints):", blocking_members if blocking_members else "None")

    # --- INDIFFERENT HOURS DETECTION ---
    if obj_cap is not None:
        indiff = dp_cap.detect_indifferent_hours()
        print(f"\n[Analysis] Detected {len(indiff)} indifferent member-hour instances under DP_cap (linear disutility effect).")

    # --- SENSITIVITY STUDY (Table 11 Requirements) ---
    print("\n=== Running Sensitivity Study on Contract Terms (Cap & Alpha) ===")
    cap_limits = [5.0, 7.5, 9.4, 12.0]
    alphas = [5.0, 10.0, 20.0]
    
    print(f"{'Cap Limit (kW)':<15} | {'Alpha Penalty':<15} | {'DP_IR Status':<15} | {'Obj Value (DKK)':<18}")
    print("-" * 68)
    
    for cap_val in cap_limits:
        for alpha_val in alphas:
            sens_model = DynamicPricingModel(members_data, contract_data, day_data, 
                                             enable_bb=False, enable_ir=True, 
                                             cap_override=cap_val, alpha_override=alpha_val)
            sens_model.build_model(ref_costs_I=ref_costs_I)
            _, sens_obj = sens_model.solve()
            
            status_str = "Optimal" if sens_obj is not None else "Infeasible"
            obj_str = f"{sens_obj:.2f} DKK" if sens_obj is not None else "N/A"
            print(f"{cap_val:<15.1f} | {alpha_val:<15.1f} | {status_str:<15} | {obj_str:<18}")

    # --- Plotting Results ---
    plt.figure(figsize=(10, 5))
    hours = list(range(24))
    
    if prices_cap:
        plt.plot(hours, prices_cap, label=r'$\mathcal{DP}^{cap}$ (Cap Only)', color='blue', linewidth=2, marker='o')
    if prices_bb:
        plt.plot(hours, prices_bb, label=r'$\mathcal{DP}^{BB}$ (Budget Balance)', color='green', linewidth=2, marker='s')
    if prices_ir:
        plt.plot(hours, prices_ir, label=r'$\mathcal{DP}^{IR}$ (Indiv. Rationality)', color='red', linewidth=2, marker='^')
        
    plt.plot(hours, day_data['p_da'], label='Day-Ahead Price ($p_t$)', color='black', linestyle='--', alpha=0.7)
    
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Internal Price (DKK/kWh)', fontsize=12)
    plt.title('Question 2: Dynamic Pricing Profiles Comparison', fontsize=14, fontweight='bold')
    plt.xticks(hours)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plot_filename = 'dp_prices_comparison.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"\n[Success] Comparison plot saved as '{plot_filename}'.")

    # --- Plotting Aggregated Community Net Power Profile ---
    def get_net_import(dp_obj):
        if dp_obj.model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return None
        return [
            dp_obj.model.getVarByName(f"P_imp[{t}]").X - dp_obj.model.getVarByName(f"P_exp[{t}]").X
            for t in dp_obj.hours
        ]

    p_net_cap = get_net_import(dp_cap)
    p_net_bb = get_net_import(dp_bb)
    p_net_ir = get_net_import(dp_ir)

    plt.figure(figsize=(10, 5))
    if p_net_cap:
        plt.plot(hours, p_net_cap, label=r'$\mathcal{DP}^{cap}$ (Cap Only)', color='blue', linewidth=2, marker='o')
    if p_net_bb:
        plt.plot(hours, p_net_bb, label=r'$\mathcal{DP}^{BB}$ (Budget Balance)', color='green', linewidth=2, marker='s')
    if p_net_ir:
        plt.plot(hours, p_net_ir, label=r'$\mathcal{DP}^{IR}$ (Indiv. Rationality)', color='red', linewidth=2, marker='^')

    plt.plot(hours, contract_data['cap'], label=r'DSO Import Cap ($\overline{P}_t^{DSO}$)', color='black', linestyle='--', alpha=0.8)
    plt.axhline(0, color='gray', linestyle=':', linewidth=1.0)

    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Net Power Import (kW)', fontsize=12)
    plt.title('Question 2: Aggregated Community Net Power Profile', fontsize=14, fontweight='bold')
    plt.xticks(hours)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    profile_filename = 'community_net_profile.png'
    plt.savefig(profile_filename, dpi=300)
    print(f"[Success] Community net profile plot saved as '{profile_filename}'.")

if __name__ == "__main__":
        main()