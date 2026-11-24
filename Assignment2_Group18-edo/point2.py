import os
import sys
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# Dynamic search for the directory containing 'src' and 'data'
current_file_dir = Path(__file__).resolve().parent
project_root = current_file_dir
while project_root != project_root.parent:
    if (project_root / "src").is_dir() and (project_root / "data").is_dir():
        break
    project_root = project_root.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_loader import load_community
from src.reference_model import ReferenceModel
from src.individual_model import IndividualModel
from src.community_model import CommunityModel


class DynamicPricingModel:
    """
    Parameterized MILP formulation of Question 2: Dynamic Pricing (DP).
    Supports:
      - Uniform hourly prices x_t (default) or per-consumer discriminated prices x_{i,t}
      - Budget Balance (enable_bb)
      - Individual Rationality (enable_ir) against per-member benchmark costs
      - Minimum IR slack relaxation (relax_ir_slack) to identify blocking members when infeasible
      - Contract parameter overrides (cap_scale, alpha_override, beta_override) for sensitivity studies
    """
    def __init__(self, community_data, contract_data, day_data,
                 enable_bb=False, enable_ir=False,
                 allow_discrimination=False, relax_ir_slack=False,
                 M=200.0, cap_scale=1.0, alpha_override=None, beta_override=None):
        self.members = community_data
        self.contract = {
            'beta': float(beta_override) if beta_override is not None else float(contract_data['beta']),
            'alpha': float(alpha_override) if alpha_override is not None else float(contract_data['alpha']),
            'cap': [float(v) * float(cap_scale) for v in contract_data['cap']],
        }
        self.day = day_data
        self.enable_bb = enable_bb
        self.enable_ir = enable_ir
        self.allow_discrimination = allow_discrimination
        self.relax_ir_slack = relax_ir_slack
        self.M = M

        self.hours = list(range(24))
        self.model = gp.Model("DynamicPricing_DP")
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('TimeLimit', 60)
        self.model.setParam('MIPGap', 1e-4)

    def build_model(self, ir_bounds=None):
        m = self.model
        H = self.hours
        I = self.members
        n_members = len(I)

        # 1. Upper-level decision variables (Community Manager)
        if self.allow_discrimination:
            x = m.addVars(n_members, H, lb=-50.0, ub=50.0, name="x_price")
        else:
            x_uniform = m.addVars(H, lb=-50.0, ub=50.0, name="x_price")
            x = {(i_idx, t): x_uniform[t] for i_idx in range(n_members) for t in H}

        P_imp = m.addVars(H, lb=0.0, name="P_imp")
        P_exp = m.addVars(H, lb=0.0, name="P_exp")
        E_exc = m.addVars(H, lb=0.0, name="E_exc")

        # Optional non-negative IR slack variables to identify blocking members
        ir_slack = m.addVars(n_members, lb=0.0, name="ir_slack") if self.relax_ir_slack else {i_idx: 0.0 for i_idx in range(n_members)}

        # 2. Lower-level primal variables
        q, l, delta = {}, {}, {}
        p_ch, p_dis, s, g = {}, {}, {}, {}

        for i_idx, i in enumerate(I):
            for t in H:
                q[i_idx, t] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"q_{i_idx}_{t}")
                l[i_idx, t] = m.addVar(lb=i['l_min'][t], ub=i['l_max'][t], name=f"l_{i_idx}_{t}")
                delta[i_idx, t] = m.addVar(lb=0.0, name=f"delta_{i_idx}_{t}")

                if i['has_battery']:
                    p_ch[i_idx, t] = m.addVar(lb=0.0, ub=i['p_ch_max'], name=f"p_ch_{i_idx}_{t}")
                    p_dis[i_idx, t] = m.addVar(lb=0.0, ub=i['p_dis_max'], name=f"p_dis_{i_idx}_{t}")
                    s[i_idx, t] = m.addVar(lb=i['s_min'], ub=i['s_max'], name=f"s_{i_idx}_{t}")
                else:
                    p_ch[i_idx, t], p_dis[i_idx, t], s[i_idx, t] = 0.0, 0.0, 0.0

                if i['has_pv']:
                    g[i_idx, t] = m.addVar(lb=0.0, ub=i['pv_max'][t], name=f"g_{i_idx}_{t}")
                else:
                    g[i_idx, t] = 0.0

        # 3. KKT Dual & Binary variables for Big-M linearization
        mu_dev_pos, mu_dev_neg, z_dev_pos, z_dev_neg = {}, {}, {}, {}
        mu_l_min, mu_l_max, z_l_min, z_l_max = {}, {}, {}, {}
        mu_energy, z_energy = {}, {}
        mu_g_min, mu_g_max, z_g_min, z_g_max = {}, {}, {}, {}
        lambda_soc = {}
        mu_ch_min, mu_ch_max, z_ch_min, z_ch_max = {}, {}, {}, {}
        mu_dis_min, mu_dis_max, z_dis_min, z_dis_max = {}, {}, {}, {}
        mu_s_min, mu_s_max, z_s_min, z_s_max = {}, {}, {}, {}

        for i_idx, i in enumerate(I):
            mu_energy[i_idx] = m.addVar(lb=0.0, name=f"mu_energy_{i_idx}")
            z_energy[i_idx] = m.addVar(vtype=GRB.BINARY, name=f"z_energy_{i_idx}")

            for t in H:
                mu_dev_pos[i_idx, t] = m.addVar(lb=0.0, name=f"mu_dev_pos_{i_idx}_{t}")
                mu_dev_neg[i_idx, t] = m.addVar(lb=0.0, name=f"mu_dev_neg_{i_idx}_{t}")
                z_dev_pos[i_idx, t] = m.addVar(vtype=GRB.BINARY, name=f"z_dev_pos_{i_idx}_{t}")
                z_dev_neg[i_idx, t] = m.addVar(vtype=GRB.BINARY, name=f"z_dev_neg_{i_idx}_{t}")

                mu_l_min[i_idx, t] = m.addVar(lb=0.0, name=f"mu_l_min_{i_idx}_{t}")
                mu_l_max[i_idx, t] = m.addVar(lb=0.0, name=f"mu_l_max_{i_idx}_{t}")
                z_l_min[i_idx, t] = m.addVar(vtype=GRB.BINARY, name=f"z_l_min_{i_idx}_{t}")
                z_l_max[i_idx, t] = m.addVar(vtype=GRB.BINARY, name=f"z_l_max_{i_idx}_{t}")

                if i['has_pv']:
                    mu_g_min[i_idx, t] = m.addVar(lb=0.0, name=f"mu_g_min_{i_idx}_{t}")
                    mu_g_max[i_idx, t] = m.addVar(lb=0.0, name=f"mu_g_max_{i_idx}_{t}")
                    z_g_min[i_idx, t] = m.addVar(vtype=GRB.BINARY, name=f"z_g_min_{i_idx}_{t}")
                    z_g_max[i_idx, t] = m.addVar(vtype=GRB.BINARY, name=f"z_g_max_{i_idx}_{t}")

                if i['has_battery']:
                    lambda_soc[i_idx, t] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"lam_soc_{i_idx}_{t}")
                    mu_ch_min[i_idx, t] = m.addVar(lb=0.0, name=f"mu_ch_min_{i_idx}_{t}")
                    mu_ch_max[i_idx, t] = m.addVar(lb=0.0, name=f"mu_ch_max_{i_idx}_{t}")
                    z_ch_min[i_idx, t] = m.addVar(vtype=GRB.BINARY, name=f"z_ch_min_{i_idx}_{t}")
                    z_ch_max[i_idx, t] = m.addVar(vtype=GRB.BINARY, name=f"z_ch_max_{i_idx}_{t}")

                    mu_dis_min[i_idx, t] = m.addVar(lb=0.0, name=f"mu_dis_min_{i_idx}_{t}")
                    mu_dis_max[i_idx, t] = m.addVar(lb=0.0, name=f"mu_dis_max_{i_idx}_{t}")
                    z_dis_min[i_idx, t] = m.addVar(vtype=GRB.BINARY, name=f"z_dis_min_{i_idx}_{t}")
                    z_dis_max[i_idx, t] = m.addVar(vtype=GRB.BINARY, name=f"z_dis_max_{i_idx}_{t}")

                    mu_s_min[i_idx, t] = m.addVar(lb=0.0, name=f"mu_s_min_{i_idx}_{t}")
                    mu_s_max[i_idx, t] = m.addVar(lb=0.0, name=f"mu_s_max_{i_idx}_{t}")
                    z_s_min[i_idx, t] = m.addVar(vtype=GRB.BINARY, name=f"z_s_min_{i_idx}_{t}")
                    z_s_max[i_idx, t] = m.addVar(vtype=GRB.BINARY, name=f"z_s_max_{i_idx}_{t}")

        # 4. Upper-level grid balance and capacity constraints
        for t in H:
            m.addConstr(gp.quicksum(q[i_idx, t] for i_idx in range(n_members)) == P_imp[t] - P_exp[t], name=f"grid_bal_{t}")
            m.addConstr(E_exc[t] >= P_imp[t] - self.contract['cap'][t], name=f"cap_exc_{t}")

        # 5. Lower-level KKT conditions per member
        for i_idx, i in enumerate(I):
            E_ref_total = i['min_daily_energy']
            m.addConstr(gp.quicksum(l[i_idx, t] for t in H) >= E_ref_total, name=f"energy_min_{i_idx}")
            m.addConstr(gp.quicksum(l[i_idx, t] for t in H) - E_ref_total <= self.M * z_energy[i_idx])
            m.addConstr(mu_energy[i_idx] <= self.M * (1 - z_energy[i_idx]))

            for t in H:
                x_it = x[i_idx, t]
                m.addConstr(q[i_idx, t] == l[i_idx, t] + p_ch[i_idx, t] - p_dis[i_idx, t] - g[i_idx, t])

                # Stationarity wrt l_{i,t} and delta_{i,t}
                m.addConstr(
                    x_it + mu_dev_pos[i_idx, t] - mu_dev_neg[i_idx, t]
                    - mu_l_min[i_idx, t] + mu_l_max[i_idx, t] - mu_energy[i_idx] == 0
                )
                m.addConstr(mu_dev_pos[i_idx, t] + mu_dev_neg[i_idx, t] <= i['c_L'])

                # Complementary slackness: load bounds
                m.addConstr(l[i_idx, t] - i['l_min'][t] <= self.M * z_l_min[i_idx, t])
                m.addConstr(mu_l_min[i_idx, t] <= self.M * (1 - z_l_min[i_idx, t]))
                m.addConstr(i['l_max'][t] - l[i_idx, t] <= self.M * z_l_max[i_idx, t])
                m.addConstr(mu_l_max[i_idx, t] <= self.M * (1 - z_l_max[i_idx, t]))

                # Complementary slackness: positive & negative deviations
                m.addConstr(l[i_idx, t] - i['l_ref'][t] - delta[i_idx, t] <= 0)
                m.addConstr(delta[i_idx, t] - l[i_idx, t] + i['l_ref'][t] <= self.M * z_dev_pos[i_idx, t])
                m.addConstr(mu_dev_pos[i_idx, t] <= self.M * (1 - z_dev_pos[i_idx, t]))

                m.addConstr(i['l_ref'][t] - l[i_idx, t] - delta[i_idx, t] <= 0)
                m.addConstr(delta[i_idx, t] - i['l_ref'][t] + l[i_idx, t] <= self.M * z_dev_neg[i_idx, t])
                m.addConstr(mu_dev_neg[i_idx, t] <= self.M * (1 - z_dev_neg[i_idx, t]))

                # PV KKT conditions
                if i['has_pv']:
                    m.addConstr(i['c_pv'] - x_it - mu_g_min[i_idx, t] + mu_g_max[i_idx, t] == 0)
                    m.addConstr(g[i_idx, t] <= self.M * z_g_min[i_idx, t])
                    m.addConstr(mu_g_min[i_idx, t] <= self.M * (1 - z_g_min[i_idx, t]))
                    m.addConstr(i['pv_max'][t] - g[i_idx, t] <= self.M * z_g_max[i_idx, t])
                    m.addConstr(mu_g_max[i_idx, t] <= self.M * (1 - z_g_max[i_idx, t]))

                # Battery KKT conditions (cyclic SOC s[23] == s_init)
                if i['has_battery']:
                    eta_ch = i['eta_ch']
                    eta_dis = i['eta_dis']
                    s_init = i['s_init']

                    if t == 0:
                        m.addConstr(s[i_idx, t] == s_init + eta_ch * p_ch[i_idx, t] - (1.0 / eta_dis) * p_dis[i_idx, t])
                    else:
                        m.addConstr(s[i_idx, t] == s[i_idx, t - 1] + eta_ch * p_ch[i_idx, t] - (1.0 / eta_dis) * p_dis[i_idx, t])

                    if t == 23:
                        m.addConstr(s[i_idx, t] == s_init)

                    m.addConstr(x_it - eta_ch * lambda_soc[i_idx, t] - mu_ch_min[i_idx, t] + mu_ch_max[i_idx, t] == 0)
                    m.addConstr(-x_it + (1.0 / eta_dis) * lambda_soc[i_idx, t] - mu_dis_min[i_idx, t] + mu_dis_max[i_idx, t] == 0)

                    if t < 23:
                        m.addConstr(lambda_soc[i_idx, t] - lambda_soc[i_idx, t + 1] - mu_s_min[i_idx, t] + mu_s_max[i_idx, t] == 0)

                    m.addConstr(p_ch[i_idx, t] <= self.M * z_ch_min[i_idx, t])
                    m.addConstr(mu_ch_min[i_idx, t] <= self.M * (1 - z_ch_min[i_idx, t]))
                    m.addConstr(i['p_ch_max'] - p_ch[i_idx, t] <= self.M * z_ch_max[i_idx, t])
                    m.addConstr(mu_ch_max[i_idx, t] <= self.M * (1 - z_ch_max[i_idx, t]))

                    m.addConstr(p_dis[i_idx, t] <= self.M * z_dis_min[i_idx, t])
                    m.addConstr(mu_dis_min[i_idx, t] <= self.M * (1 - z_dis_min[i_idx, t]))
                    m.addConstr(i['p_dis_max'] - p_dis[i_idx, t] <= self.M * z_dis_max[i_idx, t])
                    m.addConstr(mu_dis_max[i_idx, t] <= self.M * (1 - z_dis_max[i_idx, t]))

                    m.addConstr(s[i_idx, t] - i['s_min'] <= self.M * z_s_min[i_idx, t])
                    m.addConstr(mu_s_min[i_idx, t] <= self.M * (1 - z_s_min[i_idx, t]))
                    m.addConstr(i['s_max'] - s[i_idx, t] <= self.M * z_s_max[i_idx, t])
                    m.addConstr(mu_s_max[i_idx, t] <= self.M * (1 - z_s_max[i_idx, t]))

        # Exact dual objective (= primal bill + disutility + pv_cost) per member via Strong Duality
        dual_obj = {}
        for i_idx, i in enumerate(I):
            d_i = (
                mu_energy[i_idx] * i['min_daily_energy']
                + gp.quicksum(
                    mu_dev_pos[i_idx, t] * i['l_ref'][t]
                    - mu_dev_neg[i_idx, t] * i['l_ref'][t]
                    + mu_l_min[i_idx, t] * i['l_min'][t]
                    - mu_l_max[i_idx, t] * i['l_max'][t]
                    for t in H
                )
            )
            if i['has_pv']:
                d_i -= gp.quicksum(mu_g_max[i_idx, t] * i['pv_max'][t] for t in H)
            if i['has_battery']:
                d_i += (
                    (lambda_soc[i_idx, 0] - (lambda_soc[i_idx, 23] - mu_s_min[i_idx, 23] + mu_s_max[i_idx, 23])) * i['s_init']
                    + gp.quicksum(
                        - mu_ch_max[i_idx, t] * i['p_ch_max']
                        - mu_dis_max[i_idx, t] * i['p_dis_max']
                        + mu_s_min[i_idx, t] * i['s_min']
                        - mu_s_max[i_idx, t] * i['s_max']
                        for t in H
                    )
                )
            dual_obj[i_idx] = d_i

        # 6. Budget Balance constraint (DP_BB variant) via Strong Duality
        if self.enable_bb:
            total_dso_cost_expr = gp.quicksum(
                self.day['p_da'][t] * (P_imp[t] - P_exp[t]) +
                (1.0 - self.contract['beta']) * self.day['tau_imp'] * P_imp[t] +
                (1.0 - self.contract['beta']) * self.day['tau_exp'] * P_exp[t] +
                self.contract['alpha'] * E_exc[t]
                for t in H
            )
            total_internal_revenue = gp.quicksum(
                dual_obj[i_idx]
                - gp.quicksum(
                    i['c_L'] * delta[i_idx, t] + (i['c_pv'] * g[i_idx, t] if i['has_pv'] else 0.0)
                    for t in H
                )
                for i_idx, i in enumerate(I)
            )
            m.addConstr(total_internal_revenue == total_dso_cost_expr, name="BudgetBalance")

        # 7. Individual Rationality constraint (DP_IR variant) via Strong Duality
        if self.enable_ir and ir_bounds is not None:
            for i_idx in range(n_members):
                m.addConstr(dual_obj[i_idx] <= ir_bounds[i_idx] + ir_slack[i_idx], name=f"IR_member_{i_idx}")

        # 8. Objective Function
        total_dso_cost = gp.quicksum(
            self.day['p_da'][t] * (P_imp[t] - P_exp[t]) +
            (1.0 - self.contract['beta']) * self.day['tau_imp'] * P_imp[t] +
            (1.0 - self.contract['beta']) * self.day['tau_exp'] * P_exp[t] +
            self.contract['alpha'] * E_exc[t]
            for t in H
        )
        total_member_op_cost = gp.quicksum(
            I[i_idx]['c_L'] * delta[i_idx, t] + (I[i_idx]['c_pv'] * g[i_idx, t] if I[i_idx]['has_pv'] else 0.0)
            for i_idx in range(n_members) for t in H
        )

        if self.relax_ir_slack:
            # Lexicographic-style penalty: minimize total IR slack first, then social cost
            m.setObjective(1000.0 * gp.quicksum(ir_slack[i_idx] for i_idx in range(n_members)) + total_dso_cost + total_member_op_cost, GRB.MINIMIZE)
        else:
            m.setObjective(total_dso_cost + total_member_op_cost, GRB.MINIMIZE)

    def solve(self):
        self.model.optimize()
        if self.model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and self.model.SolCount > 0:
            if self.allow_discrimination:
                posted_prices = [[self.model.getVarByName(f"x_price[{i_idx},{t}]").X for t in self.hours] for i_idx in range(len(self.members))]
            else:
                posted_prices = [self.model.getVarByName(f"x_price[{t}]").X for t in self.hours]
            return posted_prices, self.model.ObjVal
        return None, None

    def extract_summary_metrics(self):
        if self.model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT] or self.model.SolCount == 0:
            return None
        p_imp = [self.model.getVarByName(f"P_imp[{t}]").X for t in self.hours]
        p_exp = [self.model.getVarByName(f"P_exp[{t}]").X for t in self.hours]
        e_exc = [self.model.getVarByName(f"E_exc[{t}]").X for t in self.hours]

        C_cost = sum(
            self.day['p_da'][t] * (p_imp[t] - p_exp[t])
            + (1.0 - self.contract['beta']) * self.day['tau_imp'] * p_imp[t]
            + (1.0 - self.contract['beta']) * self.day['tau_exp'] * p_exp[t]
            for t in self.hours
        )
        D_cost = sum(
            self.members[i_idx]['c_L'] * self.model.getVarByName(f"delta_{i_idx}_{t}").X
            + (self.members[i_idx]['c_pv'] * self.model.getVarByName(f"g_{i_idx}_{t}").X if self.members[i_idx]['has_pv'] else 0.0)
            for i_idx in range(len(self.members)) for t in self.hours
        )
        P_cost = sum(self.contract['alpha'] * e_exc[t] for t in self.hours)
        social_cost = C_cost + D_cost + P_cost

        blocking = []
        if self.relax_ir_slack:
            for i_idx, i in enumerate(self.members):
                sv = self.model.getVarByName(f"ir_slack[{i_idx}]").X
                if sv > 1e-3:
                    blocking.append((i['id'], sv))

        return {
            'daily_import': sum(p_imp),
            'peak_import': max(p_imp),
            'exceedance': sum(e_exc),
            'C': C_cost,
            'D': D_cost,
            'P': P_cost,
            'total_cost': social_cost,
            'net_import': [p_imp[t] - p_exp[t] for t in self.hours],
            'blocking_members': blocking,
        }

    def extract_member_metrics(self, baseline_costs):
        metrics = []
        for i_idx, i in enumerate(self.members):
            if self.allow_discrimination:
                bill = sum(
                    self.model.getVarByName(f"x_price[{i_idx},{t}]").X * self.model.getVarByName(f"q_{i_idx}_{t}").X
                    for t in self.hours
                )
            else:
                bill = sum(
                    self.model.getVarByName(f"x_price[{t}]").X * self.model.getVarByName(f"q_{i_idx}_{t}").X
                    for t in self.hours
                )
            disutil = sum(
                i['c_L'] * self.model.getVarByName(f"delta_{i_idx}_{t}").X
                + (i['c_pv'] * self.model.getVarByName(f"g_{i_idx}_{t}").X if i['has_pv'] else 0.0)
                for t in self.hours
            )
            m_cost = bill + disutil
            baseline = baseline_costs[i_idx]
            metrics.append({
                'member_id': i['id'],
                'bill': bill,
                'disutility': disutil,
                'actual_cost': m_cost,
                'baseline_cost': baseline,
                'savings': baseline - m_cost
            })
        return metrics

    def detect_indifferent_hours(self):
        indifferent = []
        for i_idx, i in enumerate(self.members):
            for t in self.hours:
                mu_pos = self.model.getVarByName(f"mu_dev_pos_{i_idx}_{t}").X
                mu_neg = self.model.getVarByName(f"mu_dev_neg_{i_idx}_{t}").X
                if abs(mu_pos) < 1e-3 and abs(mu_neg) < 1e-3:
                    indifferent.append((i_idx, t))
        return indifferent


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


def main():
    print("=== Loading Question 1 Community Base Case Data (main.py structure) ===")
    com = load_community("base", day="d1")
    c = com.contract

    # Direct calls to Question 1 models (exact match with main.py)
    results_ref = ReferenceModel(com).solve()
    results_ind = IndividualModel(com).solve()
    results_com = CommunityModel(com).solve()

    total_ref_cost = float(results_ref["total_cost"])
    total_ind_cost = float(results_ind["total_cost"])
    total_comm_cost = float(results_com["total_cost"])
    ref_costs_R, ind_costs_I = compute_q1_member_costs(com, cap_scale=1.0)

    day_data = {
        'p_da': [float(x) for x in com.energy_price],
        'tau_imp': float(com.import_tariff),
        'tau_exp': float(com.export_tariff),
    }
    cap_list = [float(x) for x in c.capacity_limit_kW]
    cap_val_default = cap_list[0]
    contract_data = {
        'beta': float(c.tariff_discount_ratio),
        'alpha': float(c.penalty_DKK_per_kWh),
        'cap': cap_list,
    }

    members_data = []
    for m in com.members:
        l_ref = [float(x) for x in m.reference_load]
        l_min = [float(x) for x in m.load_lower_kWh] if m.load_lower_kWh is not None else [float(m.load_min_kWh)] * 24
        l_max = [float(x) for x in m.load_upper_kWh] if m.load_upper_kWh is not None else [float(m.load_max_kWh)] * 24
        min_daily_e = float(m.min_daily_energy_kWh) if m.min_daily_energy_kWh is not None else sum(l_ref)

        m_dict = {
            'id': m.consumer_id,
            'l_ref': l_ref,
            'min_daily_energy': min_daily_e,
            'c_L': float(m.linear_disutility),
            'l_min': l_min,
            'l_max': l_max,
            'has_pv': m.has_pv,
            'pv_max': [float(x) for x in m.pv_available] if m.has_pv else [0.0] * 24,
            'c_pv': float(m.pv_marginal_cost) if m.has_pv else 0.0,
            'has_battery': m.has_battery,
        }
        if m.has_battery:
            m_dict.update({
                'p_ch_max': float(m.battery_max_charge_kW),
                'p_dis_max': float(m.battery_max_discharge_kW),
                's_min': 0.0,
                's_max': float(m.battery_capacity_kWh),
                's_init': float(m.battery_initial_soc_kWh),
                'eta_ch': float(m.battery_charging_efficiency),
                'eta_dis': float(m.battery_discharging_efficiency),
            })
        members_data.append(m_dict)

    # =========================================================================
    # PART A: Base Case Three-Step Decomposition + Combined BB+IR & Discrimination
    # =========================================================================
    print("\n[Step 1/5] Solving DP_cap (Cap only)...")
    dp_cap = DynamicPricingModel(members_data, contract_data, day_data, enable_bb=False, enable_ir=False)
    dp_cap.build_model()
    prices_cap, _ = dp_cap.solve()
    res_cap = dp_cap.extract_summary_metrics()

    print("[Step 2/5] Solving DP_BB (Budget Balance)...")
    dp_bb = DynamicPricingModel(members_data, contract_data, day_data, enable_bb=True, enable_ir=False)
    dp_bb.build_model()
    prices_bb, _ = dp_bb.solve()
    res_bb = dp_bb.extract_summary_metrics()

    print("[Step 3/5] Solving DP_IR (Individual Rationality vs Model I)...")
    dp_ir = DynamicPricingModel(members_data, contract_data, day_data, enable_bb=False, enable_ir=True)
    dp_ir.build_model(ir_bounds=ind_costs_I)
    prices_ir, _ = dp_ir.solve()
    res_ir = dp_ir.extract_summary_metrics()

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

    print("[Step 5/5] Solving DP_BB+IR_disc (Per-consumer price discrimination x_{i,t})...")
    dp_disc = DynamicPricingModel(members_data, contract_data, day_data, enable_bb=True, enable_ir=True, allow_discrimination=True)
    dp_disc.build_model(ir_bounds=ind_costs_I)
    _, _ = dp_disc.solve()
    res_disc = dp_disc.extract_summary_metrics()

    # --- Summary Table ---
    print("\n" + "=" * 105)
    print(f"{'Model / Pricing Variant':<28} | {'Cost (DKK)':<10} | {'Daily Imp':<10} | {'Peak Imp':<9} | {'Exceed.':<8} | {'WR vs I':<8} | {'WR vs R':<8}")
    print("=" * 105)
    print(f"{'Reference (R)':<28} | {total_ref_cost:<10.2f} | {results_ref['daily_import']:<10.2f} | {results_ref['peak_import']:<9.2f} | {results_ref['exceedance']:<8.2f} | {'--':<8} | {'0.0%':<8}")
    print(f"{'Individual (I)':<28} | {total_ind_cost:<10.2f} | {results_ind['daily_import']:<10.2f} | {results_ind['peak_import']:<9.2f} | {results_ind['exceedance']:<8.2f} | {'0.0%':<8} | {'--':<8}")
    print(f"{'Centralized (C)':<28} | {total_comm_cost:<10.2f} | {results_com['daily_import']:<10.2f} | {results_com['peak_import']:<9.2f} | {results_com['exceedance']:<8.2f} | {'100.0%':<8} | {'100.0%':<8}")
    for name, res in [('DP_cap (Cap only)', res_cap),
                      ('DP_BB (Budget Balance)', res_bb),
                      ('DP_IR (Ind. Rationality)', res_ir),
                      ('DP_BB+IR (Uniform)', res_bb_ir),
                      ('DP_BB+IR (Discriminated)', res_disc)]:
        if res is not None:
            val = res['total_cost']
            wr_I = 100.0 * (total_ind_cost - val) / (total_ind_cost - total_comm_cost)
            wr_R = 100.0 * (total_ref_cost - val) / (total_ref_cost - total_comm_cost)
            print(f"{name:<28} | {val:<10.2f} | {res['daily_import']:<10.2f} | {res['peak_import']:<9.2f} | {res['exceedance']:<8.2f} | {wr_I:>6.1f}% | {wr_R:>6.1f}%")
        else:
            print(f"{name:<28} | {'Infeasible':<10} | {'--':<10} | {'--':<9} | {'--':<8} | {'N/A':<8} | {'N/A':<8}")
    print("=" * 105)

    if blocking_base:
        print("Blocking members preventing uniform DP_BB+IR:", ", ".join(f"{cid} (+{sl:.2f} DKK)" for cid, sl in blocking_base))

    # --- Per-member breakdown under DP_IR ---
    if res_ir is not None:
        print("\n=== Per-Member Metrics & Savings under DP_IR (vs Individual Contract I) ===")
        print(f"{'Member ID':<10} | {'Bill (DKK)':<12} | {'Disutil.':<10} | {'DP Cost':<12} | {'Model I':<12} | {'Savings':<10}")
        print("-" * 80)
        for m_row in dp_ir.extract_member_metrics(ind_costs_I):
            print(f"{m_row['member_id']:<10} | {m_row['bill']:<12.2f} | {m_row['disutility']:<10.2f} | "
                  f"{m_row['actual_cost']:<12.2f} | {m_row['baseline_cost']:<12.2f} | {m_row['savings']:<10.2f}")

    if res_cap is not None:
        indiff = dp_cap.detect_indifferent_hours()
        print(f"\n[Analysis] Detected {len(indiff)}/384 indifferent member-hour instances under DP_cap.")

    # =========================================================================
    # PART B: Sensitivity Study on DSO Capacity Cap (1.0x, 0.8x, 0.6x, 0.4x)
    # =========================================================================
    print("\n=== Sensitivity Study: Tightening DSO Import Cap (cap_scale) ===")
    print(f"{'Cap Scale':<10} | {'Cap (kW)':<9} | {'Model I':<9} | {'Central C':<9} | {'DP_cap':<9} | {'DP_BB':<9} | {'DP_IR':<9} | {'DP_BB+IR':<11} | {'Blocking Members (if infeasible)'}")
    print("-" * 125)

    cap_scales = [1.0, 0.8, 0.6, 0.4]
    sens_records = []
    for sc in cap_scales:
        _, ind_sc = compute_q1_member_costs(com, cap_scale=sc)
        tot_I_sc = sum(ind_sc)
        tot_C_sc = solve_centralized_scaled(members_data, contract_data, day_data, cap_scale=sc)

        m_cap = DynamicPricingModel(members_data, contract_data, day_data, enable_bb=False, enable_ir=False, cap_scale=sc)
        m_cap.build_model()
        m_cap.solve()
        r_cap = m_cap.extract_summary_metrics()

        m_bb = DynamicPricingModel(members_data, contract_data, day_data, enable_bb=True, enable_ir=False, cap_scale=sc)
        m_bb.build_model()
        m_bb.solve()
        r_bb = m_bb.extract_summary_metrics()

        m_ir = DynamicPricingModel(members_data, contract_data, day_data, enable_bb=False, enable_ir=True, cap_scale=sc)
        m_ir.build_model(ir_bounds=ind_sc)
        m_ir.solve()
        r_ir = m_ir.extract_summary_metrics()

        m_bbir = DynamicPricingModel(members_data, contract_data, day_data, enable_bb=True, enable_ir=True, cap_scale=sc)
        m_bbir.build_model(ir_bounds=ind_sc)
        m_bbir.solve()
        r_bbir = m_bbir.extract_summary_metrics()

        blk_str = "None (Feasible)"
        if r_bbir is None:
            m_rel = DynamicPricingModel(members_data, contract_data, day_data, enable_bb=True, enable_ir=True, relax_ir_slack=True, cap_scale=sc)
            m_rel.build_model(ir_bounds=ind_sc)
            m_rel.solve()
            r_rel = m_rel.extract_summary_metrics()
            if r_rel and r_rel['blocking_members']:
                blk_str = ", ".join(f"{cid}(+{sl:.1f})" for cid, sl in r_rel['blocking_members'][:4])
                if len(r_rel['blocking_members']) > 4:
                    blk_str += f" (+{len(r_rel['blocking_members']) - 4} more)"
            else:
                blk_str = "BB+IR incompatible"

        s_cap = f"{r_cap['total_cost']:.2f}" if r_cap else "Infeas."
        s_bb = f"{r_bb['total_cost']:.2f}" if r_bb else "Infeas."
        s_ir = f"{r_ir['total_cost']:.2f}" if r_ir else "Infeas."
        s_bbir = f"{r_bbir['total_cost']:.2f}" if r_bbir else "Infeasible"

        print(f"{sc:<10.1f} | {cap_val_default * sc:<9.2f} | {tot_I_sc:<9.2f} | {tot_C_sc:<9.2f} | {s_cap:<9} | {s_bb:<9} | {s_ir:<9} | {s_bbir:<11} | {blk_str}")
        sens_records.append((sc, tot_I_sc, tot_C_sc,
                             r_cap['total_cost'] if r_cap else None,
                             r_bb['total_cost'] if r_bb else None,
                             r_ir['total_cost'] if r_ir else None,
                             r_bbir['total_cost'] if r_bbir else None))

    # =========================================================================
    # PART C: Save Plots to root and figures/
    # =========================================================================
    fig_dir = project_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    hours = list(range(24))

    # Figure 1: Prices Comparison
    plt.figure(figsize=(10, 5))
    if 'dual_power_balance' in results_com:
        plt.step(hours, results_com['dual_power_balance'], where='mid',
                 label=r'Centralized Dual $\lambda_t$ ($\mathcal{C}$)', color='skyblue', linewidth=5, alpha=0.6)
    if prices_cap:
        plt.plot(hours, prices_cap, label=r'$\mathcal{DP}^{cap}$ (Cap Only)', color='blue', linewidth=2, marker='o')
    if prices_bb:
        plt.plot(hours, prices_bb, label=r'$\mathcal{DP}^{BB}$ (Budget Balance)', color='green', linewidth=2, marker='s')
    if prices_ir:
        plt.plot(hours, prices_ir, label=r'$\mathcal{DP}^{IR}$ (Indiv. Rationality)', color='red', linewidth=2, marker='^')
    if prices_bb_ir:
        plt.plot(hours, prices_bb_ir, label=r'$\mathcal{DP}^{BB+IR}$ (Combined)', color='purple', linewidth=2, linestyle='-.', marker='d')

    plt.plot(hours, day_data['p_da'], label=r'Day-Ahead Price ($p_t$)', color='black', linestyle='--', alpha=0.7)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Internal Price (DKK/kWh)', fontsize=12)
    plt.title('Question 2: Dynamic Pricing Profiles Comparison (Base Case)', fontsize=13, fontweight='bold')
    plt.xticks(hours)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('dp_prices_comparison.png', dpi=300)
    plt.savefig(fig_dir / 'dp_prices_comparison.png', dpi=300)

    # Figure 2: Net Import Profile
    plt.figure(figsize=(10, 5))
    if res_cap:
        plt.plot(hours, res_cap['net_import'], label=r'$\mathcal{DP}^{cap}$ (Cap Only)', color='blue', linewidth=2, marker='o')
    if res_bb:
        plt.plot(hours, res_bb['net_import'], label=r'$\mathcal{DP}^{BB}$ (Budget Balance)', color='green', linewidth=2, marker='s')
    if res_ir:
        plt.plot(hours, res_ir['net_import'], label=r'$\mathcal{DP}^{IR}$ (Indiv. Rationality)', color='red', linewidth=2, marker='^')
    if res_bb_ir:
        plt.plot(hours, res_bb_ir['net_import'], label=r'$\mathcal{DP}^{BB+IR}$ (Combined)', color='purple', linewidth=2, linestyle='-.', marker='d')

    plt.plot(hours, contract_data['cap'], label=rf'DSO Import Cap ($\overline{{P}}_t^{{DSO}} = {cap_val_default:.2f}$ kW)',
             color='black', linestyle='--', alpha=0.8)
    plt.axhline(0, color='gray', linestyle=':', linewidth=1.0)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Net Power Import (kW)', fontsize=12)
    plt.title('Question 2: Aggregated Community Net Power Profile (Base Case)', fontsize=13, fontweight='bold')
    plt.xticks(hours)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('community_net_profile.png', dpi=300)
    plt.savefig(fig_dir / 'community_net_profile.png', dpi=300)

    # Figure 3: Sensitivity Study Plot
    plt.figure(figsize=(9, 5))
    x_sc = [r[0] * 100 for r in sens_records]
    plt.plot(x_sc, [r[1] for r in sens_records], 'k--', marker='x', label=r'Individual Contract ($\mathcal{I}$)')
    plt.plot(x_sc, [r[2] for r in sens_records], 'b-', linewidth=3, marker='o', label=r'Centralized ($\mathcal{C}$) & $\mathcal{DP}^{cap}$')
    bb_pts = [(r[0] * 100, r[4]) for r in sens_records if r[4] is not None]
    ir_pts = [(r[0] * 100, r[5]) for r in sens_records if r[5] is not None]
    bbir_pts = [(r[0] * 100, r[6]) for r in sens_records if r[6] is not None]
    if bb_pts:
        plt.plot([p[0] for p in bb_pts], [p[1] for p in bb_pts], 'g-.', marker='s', label=r'$\mathcal{DP}^{BB}$')
    if ir_pts:
        plt.plot([p[0] for p in ir_pts], [p[1] for p in ir_pts], 'r:', marker='^', label=r'$\mathcal{DP}^{IR}$')
    if bbir_pts:
        plt.plot([p[0] for p in bbir_pts], [p[1] for p in bbir_pts], color='purple', linestyle='-', marker='d', label=r'$\mathcal{DP}^{BB+IR}$')
    plt.gca().invert_xaxis()
    plt.xlabel('DSO Import Cap (% of Base Case)', fontsize=12)
    plt.ylabel('Total Community Cost (DKK)', fontsize=12)
    plt.title('Question 2: Sensitivity to DSO Import Cap Tightening', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('dp_sensitivity_cap.png', dpi=300)
    plt.savefig(fig_dir / 'dp_sensitivity_cap.png', dpi=300)

    print(f"\n[Success] Saved all 3 figures to root and '{fig_dir}'.")


if __name__ == "__main__":
    main()