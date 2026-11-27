import os
import sys
import time
from pathlib import Path
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# Dynamic search (upwards and downwards) for the directory containing 'src' and 'data'
current_file_dir = Path(__file__).resolve().parent
project_root = None
for candidate in [current_file_dir, *current_file_dir.parents, *current_file_dir.rglob("*")]:
    if candidate.is_dir() and (candidate / "src").is_dir() and (candidate / "data").is_dir():
        project_root = candidate
        break

if project_root is not None and str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_loader import load_community


def scaled_beta_draws(rng, n_draws, lo=0.8, hi=1.2, mean=1.0, kappa=2.0):
    """
    Draws n_draws samples from a scaled Beta distribution on [lo, hi]
    parameterized by mean and concentration kappa (Table 10 rules).
    """
    if n_draws <= 1:
        return np.array([float(mean)], dtype=float)
    mu_std = (mean - lo) / (hi - lo)
    mu_std = float(np.clip(mu_std, 1e-6, 1.0 - 1e-6))
    a = mu_std * kappa
    b = (1.0 - mu_std) * kappa
    u = rng.beta(a, b, size=n_draws)
    return lo + (hi - lo) * u


def generate_joint_scenarios(members_base, n_joint, seed=42, lo=0.8, hi=1.2, mean=1.0, kappa=2.0):
    """
    Generates n_joint joint community scenarios with independent per-member
    profile scaling draws theta_{i,w} in [lo, hi].
    In each scenario:
    - l_ref_{i,t,w} = theta_{i,w} * l_ref_base_{i,t}
    - min_daily_energy_{i,w} = sum_t l_ref_{i,t,w}
    - hourly bounds are clipped around l_ref_{i,t,w} +/- delta_max within [0, L_max]
    """
    rng = np.random.default_rng(seed)
    member_thetas = {}
    for i_idx, m in enumerate(members_base):
        m_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        member_thetas[i_idx] = scaled_beta_draws(m_rng, n_joint, lo=lo, hi=hi, mean=mean, kappa=kappa)

    scenarios = []
    prob = 1.0 / max(1, n_joint)
    for w in range(n_joint):
        scen_members = []
        theta_map = {}
        for i_idx, m in enumerate(members_base):
            th = float(member_thetas[i_idx][w])
            theta_map[m['id']] = th
            l_ref_w = [th * val for val in m['l_ref']]
            delta_max = m.get('delta_max', 1.0)
            l_max_cap = m.get('l_cap_max', 3.0)
            l_min_w = [max(0.0, lr - delta_max) for lr in l_ref_w]
            l_max_w = [min(l_max_cap, lr + delta_max) for lr in l_ref_w]
            min_e_w = sum(l_ref_w)

            m_w = dict(m)
            m_w['l_ref'] = l_ref_w
            m_w['l_min'] = l_min_w
            m_w['l_max'] = l_max_w
            m_w['min_daily_energy'] = min_e_w
            scen_members.append(m_w)

        scenarios.append({
            'id': w,
            'prob': prob,
            'theta': theta_map,
            'members': scen_members
        })
    return scenarios


def expected_scenario(members_base):
    """Returns the deterministic expected scenario (theta = 1.0 for all members)."""
    return generate_joint_scenarios(members_base, n_joint=1, seed=0, lo=1.0, hi=1.0, mean=1.0, kappa=2.0)


def compute_individual_benchmarks(com, members_scen):
    """Computes per-member costs under Individual Contract (I) for a given scenario."""
    c = com.contract
    ind_costs = []
    beta_eff = float(c.tariff_discount_ratio)
    alpha_eff = float(c.penalty_DKK_per_kWh)
    p_da = [float(x) for x in com.energy_price]
    tau_imp = float(com.import_tariff)
    tau_exp = float(com.export_tariff)

    for m_obj, m_dict in zip(com.members, members_scen):
        cap_i = [float(v) for v in c.individual_capacity_limit_kW[m_obj.consumer_id]]
        rm = gp.Model()
        rm.setParam('OutputFlag', 0)
        l_r = rm.addVars(24, lb=m_dict['l_min'], ub=m_dict['l_max'])
        d_r = rm.addVars(24, lb=0.0)
        p_imp_r = rm.addVars(24, lb=0.0)
        p_exp_r = rm.addVars(24, lb=0.0)
        e_exc_r = rm.addVars(24, lb=0.0)
        g_r = rm.addVars(24, lb=0.0, ub=m_dict['pv_max']) if m_dict['has_pv'] else {t: 0.0 for t in range(24)}
        if m_dict['has_battery']:
            pch_r = rm.addVars(24, lb=0.0, ub=m_dict['p_ch_max'])
            pdis_r = rm.addVars(24, lb=0.0, ub=m_dict['p_dis_max'])
            s_r = rm.addVars(24, lb=0.0, ub=m_dict['s_max'])
        else:
            pch_r = pdis_r = s_r = {t: 0.0 for t in range(24)}

        rm.addConstr(gp.quicksum(l_r[t] for t in range(24)) >= m_dict['min_daily_energy'])
        for t in range(24):
            rm.addConstr(l_r[t] - m_dict['l_ref'][t] <= d_r[t])
            rm.addConstr(m_dict['l_ref'][t] - l_r[t] <= d_r[t])
            rm.addConstr(p_imp_r[t] - p_exp_r[t] == l_r[t] + pch_r[t] - pdis_r[t] - g_r[t])
            rm.addConstr(e_exc_r[t] >= p_imp_r[t] - cap_i[t])
            if m_dict['has_battery']:
                prev_s = m_dict['s_init'] if t == 0 else s_r[t - 1]
                rm.addConstr(s_r[t] == prev_s + m_dict['eta_ch'] * pch_r[t] - (1.0 / m_dict['eta_dis']) * pdis_r[t])
                if t == 23:
                    rm.addConstr(s_r[t] == m_dict['s_init'])

        rm.setObjective(
            gp.quicksum(
                p_da[t] * (p_imp_r[t] - p_exp_r[t])
                + (1.0 - beta_eff) * tau_imp * p_imp_r[t]
                + (1.0 - beta_eff) * tau_exp * p_exp_r[t]
                + alpha_eff * e_exc_r[t]
                + m_dict['c_L'] * d_r[t]
                + (m_dict['c_pv'] * g_r[t] if m_dict['has_pv'] else 0.0)
                for t in range(24)
            ),
            GRB.MINIMIZE,
        )
        rm.optimize()
        ind_costs.append(float(rm.ObjVal))
    return ind_costs


class StochasticDynamicPricingModel:
    """
    Parameterized Scenario-Based Stochastic Bilevel MILP (SDP^x) for Question 3.
    Aligned with Question 2 KKT formulation:
    - First-stage uniform hourly price vector x_t posted day-ahead
    - Scenario-indexed lower-level primal & dual KKT systems for w in {0, ..., N-1}
    - Supports Risk-Neutral (risk_weight=0), CVaR tail hedging (risk_weight>0),
      Expected Budget Balance, Expected Individual Rationality, and
      Joint Chance-Constrained Cap Protection (cap_mode='chance').
    """
    def __init__(self, scenarios, contract_data, day_data,
                 bb_mode='expectation', ir_mode='none',
                 ir_bounds_per_scen=None,
                 risk_weight=0.0, cvar_alpha=0.80,
                 cap_mode='penalty', eps_cap=0.20,
                 M=200.0, time_limit=60, mip_gap=1e-3):
        self.scenarios = scenarios
        self.n_scen = len(scenarios)
        self.n_members = len(scenarios[0]['members'])
        self.contract = contract_data
        self.day = day_data
        self.bb_mode = bb_mode
        self.ir_mode = ir_mode
        self.ir_bounds_per_scen = ir_bounds_per_scen
        self.risk_weight = float(risk_weight)
        self.cvar_alpha = float(cvar_alpha)
        self.cap_mode = cap_mode
        self.eps_cap = float(eps_cap)
        self.M = float(M)

        self.hours = list(range(24))
        self.model = gp.Model("StochasticDynamicPricing_SDP")
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('TimeLimit', time_limit)
        self.model.setParam('MIPGap', mip_gap)
        self.model.setParam('MIPFocus', 1)
        self.model.setParam('Heuristics', 0.25)

    def build_model(self):
        m = self.model
        H = self.hours
        W = range(self.n_scen)
        n_m = self.n_members

        # 1. First-stage uniform hourly prices posted day-ahead
        x = m.addVars(H, lb=-50.0, ub=50.0, name="x_price")

        # 2. Second-stage grid interface variables per scenario
        P_imp = m.addVars(W, H, lb=0.0, name="P_imp")
        P_exp = m.addVars(W, H, lb=0.0, name="P_exp")
        E_exc = m.addVars(W, H, lb=0.0, name="E_exc")

        # Chance-constraint binary indicator per scenario (1 = violation allowed)
        y_viol = m.addVars(W, vtype=GRB.BINARY, name="y_viol") if self.cap_mode == 'chance' else None

        # CVaR auxiliary variables
        eta_cvar = m.addVar(lb=-GRB.INFINITY, name="eta_cvar")
        u_cvar = m.addVars(W, lb=0.0, name="u_cvar")

        q, l, delta, p_ch, p_dis, s, g = {}, {}, {}, {}, {}, {}, {}
        scen_dso_cost = {}
        scen_total_cost = {}
        scen_member_total_cost = {}
        scen_member_bill = {}

        for w in W:
            members_w = self.scenarios[w]['members']

            for i_idx, i in enumerate(members_w):
                mu_energy = m.addVar(lb=0.0)
                z_energy = m.addVar(vtype=GRB.BINARY)

                mu_dev_pos = m.addVars(H, lb=0.0)
                mu_dev_neg = m.addVars(H, lb=0.0)
                z_dev_pos = m.addVars(H, vtype=GRB.BINARY)
                z_dev_neg = m.addVars(H, vtype=GRB.BINARY)

                mu_l_min = m.addVars(H, lb=0.0)
                mu_l_max = m.addVars(H, lb=0.0)
                z_l_min = m.addVars(H, vtype=GRB.BINARY)
                z_l_max = m.addVars(H, vtype=GRB.BINARY)

                if i['has_pv']:
                    mu_g_min = m.addVars(H, lb=0.0)
                    mu_g_max = m.addVars(H, lb=0.0)
                    z_g_min = m.addVars(H, vtype=GRB.BINARY)
                    z_g_max = m.addVars(H, vtype=GRB.BINARY)

                if i['has_battery']:
                    lam_soc = m.addVars(H, lb=-GRB.INFINITY, ub=GRB.INFINITY)
                    mu_ch_min = m.addVars(H, lb=0.0)
                    mu_ch_max = m.addVars(H, lb=0.0)
                    z_ch_min = m.addVars(H, vtype=GRB.BINARY)
                    z_ch_max = m.addVars(H, vtype=GRB.BINARY)

                    mu_dis_min = m.addVars(H, lb=0.0)
                    mu_dis_max = m.addVars(H, lb=0.0)
                    z_dis_min = m.addVars(H, vtype=GRB.BINARY)
                    z_dis_max = m.addVars(H, vtype=GRB.BINARY)

                    mu_s_min = m.addVars(H, lb=0.0)
                    mu_s_max = m.addVars(H, lb=0.0)
                    z_s_min = m.addVars(H, vtype=GRB.BINARY)
                    z_s_max = m.addVars(H, vtype=GRB.BINARY)

                for t in H:
                    q[w, i_idx, t] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
                    l[w, i_idx, t] = m.addVar(lb=i['l_min'][t], ub=i['l_max'][t])
                    delta[w, i_idx, t] = m.addVar(lb=0.0)
                    g[w, i_idx, t] = m.addVar(lb=0.0, ub=i['pv_max'][t]) if i['has_pv'] else 0.0

                    if i['has_battery']:
                        p_ch[w, i_idx, t] = m.addVar(lb=0.0, ub=i['p_ch_max'])
                        p_dis[w, i_idx, t] = m.addVar(lb=0.0, ub=i['p_dis_max'])
                        s[w, i_idx, t] = m.addVar(lb=i['s_min'], ub=i['s_max'])
                    else:
                        p_ch[w, i_idx, t] = p_dis[w, i_idx, t] = s[w, i_idx, t] = 0.0

                # Daily minimum energy KKT
                E_min_w = i['min_daily_energy']
                m.addConstr(gp.quicksum(l[w, i_idx, t] for t in H) >= E_min_w)
                m.addConstr(gp.quicksum(l[w, i_idx, t] for t in H) - E_min_w <= self.M * z_energy)
                m.addConstr(mu_energy <= self.M * (1 - z_energy))

                for t in H:
                    m.addConstr(q[w, i_idx, t] == l[w, i_idx, t] + p_ch[w, i_idx, t] - p_dis[w, i_idx, t] - g[w, i_idx, t])
                    m.addConstr(
                        x[t] + mu_dev_pos[t] - mu_dev_neg[t]
                        - mu_l_min[t] + mu_l_max[t] - mu_energy == 0
                    )
                    m.addConstr(mu_dev_pos[t] + mu_dev_neg[t] <= i['c_L'])

                    m.addConstr(l[w, i_idx, t] - i['l_min'][t] <= self.M * z_l_min[t])
                    m.addConstr(mu_l_min[t] <= self.M * (1 - z_l_min[t]))
                    m.addConstr(i['l_max'][t] - l[w, i_idx, t] <= self.M * z_l_max[t])
                    m.addConstr(mu_l_max[t] <= self.M * (1 - z_l_max[t]))

                    m.addConstr(l[w, i_idx, t] - i['l_ref'][t] - delta[w, i_idx, t] <= 0)
                    m.addConstr(delta[w, i_idx, t] - l[w, i_idx, t] + i['l_ref'][t] <= self.M * z_dev_pos[t])
                    m.addConstr(mu_dev_pos[t] <= self.M * (1 - z_dev_pos[t]))

                    m.addConstr(i['l_ref'][t] - l[w, i_idx, t] - delta[w, i_idx, t] <= 0)
                    m.addConstr(delta[w, i_idx, t] - i['l_ref'][t] + l[w, i_idx, t] <= self.M * z_dev_neg[t])
                    m.addConstr(mu_dev_neg[t] <= self.M * (1 - z_dev_neg[t]))

                    if i['has_pv']:
                        m.addConstr(i['c_pv'] - x[t] - mu_g_min[t] + mu_g_max[t] == 0)
                        m.addConstr(g[w, i_idx, t] <= self.M * z_g_min[t])
                        m.addConstr(mu_g_min[t] <= self.M * (1 - z_g_min[t]))
                        m.addConstr(i['pv_max'][t] - g[w, i_idx, t] <= self.M * z_g_max[t])
                        m.addConstr(mu_g_max[t] <= self.M * (1 - z_g_max[t]))

                    if i['has_battery']:
                        eta_ch = i['eta_ch']
                        eta_dis = i['eta_dis']
                        s_init = i['s_init']
                        prev_s = s_init if t == 0 else s[w, i_idx, t - 1]
                        m.addConstr(s[w, i_idx, t] == prev_s + eta_ch * p_ch[w, i_idx, t] - (1.0 / eta_dis) * p_dis[w, i_idx, t])
                        if t == 23:
                            m.addConstr(s[w, i_idx, t] == s_init)

                        # KKT stationarity aligned with point2.py
                        m.addConstr(x[t] - eta_ch * lam_soc[t] - mu_ch_min[t] + mu_ch_max[t] == 0)
                        m.addConstr(-x[t] + (1.0 / eta_dis) * lam_soc[t] - mu_dis_min[t] + mu_dis_max[t] == 0)
                        if t < 23:
                            m.addConstr(lam_soc[t] - lam_soc[t + 1] - mu_s_min[t] + mu_s_max[t] == 0)

                        m.addConstr(p_ch[w, i_idx, t] <= self.M * z_ch_min[t])
                        m.addConstr(mu_ch_min[t] <= self.M * (1 - z_ch_min[t]))
                        m.addConstr(i['p_ch_max'] - p_ch[w, i_idx, t] <= self.M * z_ch_max[t])
                        m.addConstr(mu_ch_max[t] <= self.M * (1 - z_ch_max[t]))

                        m.addConstr(p_dis[w, i_idx, t] <= self.M * z_dis_min[t])
                        m.addConstr(mu_dis_min[t] <= self.M * (1 - z_dis_min[t]))
                        m.addConstr(i['p_dis_max'] - p_dis[w, i_idx, t] <= self.M * z_dis_max[t])
                        m.addConstr(mu_dis_max[t] <= self.M * (1 - z_dis_max[t]))

                        m.addConstr(s[w, i_idx, t] - i['s_min'] <= self.M * z_s_min[t])
                        m.addConstr(mu_s_min[t] <= self.M * (1 - z_s_min[t]))
                        m.addConstr(i['s_max'] - s[w, i_idx, t] <= self.M * z_s_max[t])
                        m.addConstr(mu_s_max[t] <= self.M * (1 - z_s_max[t]))

                # Dual objective aligned with point2.py
                d_wi = (
                    mu_energy * E_min_w
                    + gp.quicksum(
                        mu_dev_pos[t] * i['l_ref'][t] - mu_dev_neg[t] * i['l_ref'][t]
                        + mu_l_min[t] * i['l_min'][t] - mu_l_max[t] * i['l_max'][t]
                        for t in H
                    )
                )
                if i['has_pv']:
                    d_wi -= gp.quicksum(mu_g_max[t] * i['pv_max'][t] for t in H)
                if i['has_battery']:
                    d_wi += (
                        (lam_soc[0] - (lam_soc[23] - mu_s_min[23] + mu_s_max[23])) * i['s_init']
                        + gp.quicksum(
                            -mu_ch_max[t] * i['p_ch_max'] - mu_dis_max[t] * i['p_dis_max']
                            + mu_s_min[t] * i['s_min'] - mu_s_max[t] * i['s_max']
                            for t in H
                        )
                    )
                op_wi = gp.quicksum(
                    i['c_L'] * delta[w, i_idx, t] + (i['c_pv'] * g[w, i_idx, t] if i['has_pv'] else 0.0)
                    for t in H
                )
                scen_member_total_cost[w, i_idx] = d_wi
                scen_member_bill[w, i_idx] = d_wi - op_wi

            # Upper-level grid balance & cap constraints in scenario w
            for t in H:
                m.addConstr(gp.quicksum(q[w, i_idx, t] for i_idx in range(n_m)) == P_imp[w, t] - P_exp[w, t])
                m.addConstr(E_exc[w, t] >= P_imp[w, t] - self.contract['cap'][t])

            if self.cap_mode == 'chance':
                # Big-M bound for daily exceedance when y_viol[w] == 1
                m.addConstr(gp.quicksum(E_exc[w, t] for t in H) <= self.M * y_viol[w])

            dso_w = gp.quicksum(
                self.day['p_da'][t] * (P_imp[w, t] - P_exp[w, t])
                + (1.0 - self.contract['beta']) * self.day['tau_imp'] * P_imp[w, t]
                + (1.0 - self.contract['beta']) * self.day['tau_exp'] * P_exp[w, t]
                + self.contract['alpha'] * E_exc[w, t]
                for t in H
            )
            op_w = gp.quicksum(
                members_w[i_idx]['c_L'] * delta[w, i_idx, t]
                + (members_w[i_idx]['c_pv'] * g[w, i_idx, t] if members_w[i_idx]['has_pv'] else 0.0)
                for i_idx in range(n_m) for t in H
            )
            scen_dso_cost[w] = dso_w
            scen_total_cost[w] = dso_w + op_w

            # CVaR epigraph constraint per scenario
            m.addConstr(u_cvar[w] >= scen_total_cost[w] - eta_cvar)

        # 4. Expected Budget Balance & Individual Rationality with practical tolerance
        if self.bb_mode == 'expectation':
            exp_rev = gp.quicksum(
                self.scenarios[w]['prob'] * gp.quicksum(scen_member_bill[w, i_idx] for i_idx in range(n_m))
                for w in W
            )
            exp_dso = gp.quicksum(self.scenarios[w]['prob'] * scen_dso_cost[w] for w in W)
            bb_tol = 0.25
            m.addConstr(exp_rev - exp_dso <= bb_tol, name="Expected_BudgetBalance_UB")
            m.addConstr(exp_rev - exp_dso >= -bb_tol, name="Expected_BudgetBalance_LB")

        if self.ir_mode == 'expectation' and self.ir_bounds_per_scen is not None:
            for i_idx in range(n_m):
                exp_m_cost = gp.quicksum(self.scenarios[w]['prob'] * scen_member_total_cost[w, i_idx] for w in W)
                exp_ir_bound = sum(self.scenarios[w]['prob'] * self.ir_bounds_per_scen[w][i_idx] for w in W)
                m.addConstr(exp_m_cost <= exp_ir_bound + 0.25, name=f"Expected_IR_{i_idx}")

        if self.cap_mode == 'chance':
            m.addConstr(
                gp.quicksum(self.scenarios[w]['prob'] * y_viol[w] for w in W) <= self.eps_cap,
                name="Cap_ChanceConstraint"
            )

        # 5. Objective: (1 - lambda) * E[Total Cost] + lambda * CVaR_alpha(Total Cost)
        expected_cost = gp.quicksum(self.scenarios[w]['prob'] * scen_total_cost[w] for w in W)
        cvar_cost = eta_cvar + (1.0 / (1.0 - self.cvar_alpha)) * gp.quicksum(
            self.scenarios[w]['prob'] * u_cvar[w] for w in W
        )
        m.setObjective((1.0 - self.risk_weight) * expected_cost + self.risk_weight * cvar_cost, GRB.MINIMIZE)

    def solve(self):
        t0 = time.perf_counter()
        self.model.optimize()
        elapsed = time.perf_counter() - t0
        if self.model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and self.model.SolCount > 0:
            prices = [self.model.getVarByName(f"x_price[{t}]").X for t in self.hours]
            return prices, float(self.model.ObjVal), elapsed, self.model.status
        return None, None, elapsed, self.model.status


def evaluate_prices_on_scenarios(posted_prices, scenarios, contract_data, day_data, ir_bounds_per_scen=None):
    """
    Exact out-of-sample / in-sample simulation: given fixed posted hourly prices x_t,
    solves each member's optimal response in each scenario w (with optimistic tie-breaking
    for indifferent schedules), then computes realized grid imports, cap exceedance,
    social cost, BB gap, and IR violations.
    """
    H = range(24)
    n_m = len(scenarios[0]['members'])
    scen_metrics = []

    for w_idx, sc in enumerate(scenarios):
        members_w = sc['members']
        q_tot = [0.0] * 24
        op_cost_w = 0.0
        member_bills_w = []
        member_costs_w = []

        for i_idx, i in enumerate(members_w):
            # Step A: find minimum cost for member i under posted_prices
            lp = gp.Model()
            lp.setParam('OutputFlag', 0)
            l_v = lp.addVars(H, lb=i['l_min'], ub=i['l_max'])
            d_v = lp.addVars(H, lb=0.0)
            g_v = lp.addVars(H, lb=0.0, ub=i['pv_max']) if i['has_pv'] else {t: 0.0 for t in H}
            if i['has_battery']:
                pch_v = lp.addVars(H, lb=0.0, ub=i['p_ch_max'])
                pdis_v = lp.addVars(H, lb=0.0, ub=i['p_dis_max'])
                s_v = lp.addVars(H, lb=i['s_min'], ub=i['s_max'])
            else:
                pch_v = pdis_v = s_v = {t: 0.0 for t in H}

            lp.addConstr(gp.quicksum(l_v[t] for t in H) >= i['min_daily_energy'])
            for t in H:
                lp.addConstr(l_v[t] - i['l_ref'][t] <= d_v[t])
                lp.addConstr(i['l_ref'][t] - l_v[t] <= d_v[t])
                if i['has_battery']:
                    prev_s = i['s_init'] if t == 0 else s_v[t - 1]
                    lp.addConstr(s_v[t] == prev_s + i['eta_ch'] * pch_v[t] - (1.0 / i['eta_dis']) * pdis_v[t])
                    if t == 23:
                        lp.addConstr(s_v[t] == i['s_init'])

            obj_expr = gp.quicksum(
                posted_prices[t] * (l_v[t] + pch_v[t] - pdis_v[t] - g_v[t])
                + i['c_L'] * d_v[t]
                + (i['c_pv'] * g_v[t] if i['has_pv'] else 0.0)
                for t in H
            )
            lp.setObjective(obj_expr, GRB.MINIMIZE)
            lp.optimize()
            opt_member_cost = float(lp.ObjVal)

            # Step B: optimistic tie-breaking among equally optimal member schedules
            # (minimizes deviation from reference profile when indifferent)
            lp.addConstr(obj_expr <= opt_member_cost + 1e-5)
            lp.setObjective(gp.quicksum(d_v[t] for t in H), GRB.MINIMIZE)
            lp.optimize()

            bill_i = 0.0
            op_i = 0.0
            for t in H:
                lt = float(l_v[t].X)
                dt = float(d_v[t].X)
                gt = float(g_v[t].X) if i['has_pv'] else 0.0
                pcht = float(pch_v[t].X) if i['has_battery'] else 0.0
                pdist = float(pdis_v[t].X) if i['has_battery'] else 0.0
                qt = lt + pcht - pdist - gt
                q_tot[t] += qt
                bill_i += posted_prices[t] * qt
                op_i += i['c_L'] * dt + (i['c_pv'] * gt if i['has_pv'] else 0.0)

            op_cost_w += op_i
            member_bills_w.append(bill_i)
            member_costs_w.append(bill_i + op_i)

        p_imp_w = [max(0.0, q_tot[t]) for t in H]
        p_exp_w = [max(0.0, -q_tot[t]) for t in H]
        e_exc_w = [max(0.0, p_imp_w[t] - contract_data['cap'][t]) for t in H]

        dso_cost_w = sum(
            day_data['p_da'][t] * (p_imp_w[t] - p_exp_w[t])
            + (1.0 - contract_data['beta']) * day_data['tau_imp'] * p_imp_w[t]
            + (1.0 - contract_data['beta']) * day_data['tau_exp'] * p_exp_w[t]
            + contract_data['alpha'] * e_exc_w[t]
            for t in H
        )
        social_cost_w = dso_cost_w + op_cost_w
        bb_deficit_w = dso_cost_w - sum(member_bills_w)
        ir_viol_w = 0.0
        if ir_bounds_per_scen is not None:
            ir_viol_w = sum(max(0.0, member_costs_w[i_idx] - ir_bounds_per_scen[w_idx][i_idx]) for i_idx in range(n_m))

        scen_metrics.append({
            'social_cost': social_cost_w,
            'dso_cost': dso_cost_w,
            'op_cost': op_cost_w,
            'exceedance_kwh': sum(e_exc_w),
            'violated_cap': 1 if sum(e_exc_w) > 1e-3 else 0,
            'bb_deficit': bb_deficit_w,
            'ir_violation': ir_viol_w,
        })

    costs = np.array([r['social_cost'] for r in scen_metrics])
    excs = np.array([r['exceedance_kwh'] for r in scen_metrics])
    cap_viols = np.array([r['violated_cap'] for r in scen_metrics])
    bb_defs = np.array([r['bb_deficit'] for r in scen_metrics])
    ir_viols = np.array([r['ir_violation'] for r in scen_metrics])

    q90 = float(np.quantile(costs, 0.90))
    tail_costs = costs[costs >= q90]
    cvar90 = float(np.mean(tail_costs)) if len(tail_costs) > 0 else float(np.max(costs))

    return {
        'costs': costs,
        'mean_cost': float(np.mean(costs)),
        'std_cost': float(np.std(costs)),
        'q90_cost': q90,
        'cvar90_cost': cvar90,
        'max_cost': float(np.max(costs)),
        'cap_viol_prob': float(np.mean(cap_viols) * 100.0),
        'mean_exc_kwh': float(np.mean(excs)),
        'mean_bb_deficit': float(np.mean(bb_defs)),
        'ir_viol_prob': float(np.mean(ir_viols > 1e-3) * 100.0),
        'mean_ir_viol': float(np.mean(ir_viols)),
    }


def main():
    print("=== Question 3: Stochastic Dynamic Pricing under Preference Uncertainty ===")
    com = load_community("base", day="d1")
    c = com.contract

    day_data = {
        'p_da': [float(x) for x in com.energy_price],
        'tau_imp': float(com.import_tariff),
        'tau_exp': float(com.export_tariff),
    }
    contract_data = {
        'beta': float(c.tariff_discount_ratio),
        'alpha': float(c.penalty_DKK_per_kWh),
        'cap': [float(x) for x in c.capacity_limit_kW],
    }

    members_base = []
    for m in com.members:
        l_ref = [float(x) for x in m.reference_load]
        l_min = [float(x) for x in m.load_lower_kWh] if m.load_lower_kWh is not None else [float(m.load_min_kWh)] * 24
        l_max = [float(x) for x in m.load_upper_kWh] if m.load_upper_kWh is not None else [float(m.load_max_kWh)] * 24
        m_dict = {
            'id': m.consumer_id,
            'l_ref': l_ref,
            'min_daily_energy': sum(l_ref),
            'delta_max': float(getattr(m, 'max_hourly_deviation_kWh', 1.0)),
            'l_cap_max': float(getattr(m, 'load_max_kWh', 3.0)),
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
        members_base.append(m_dict)

    # 1. Generate reproducible In-Sample and Out-of-Sample scenario sets (Table 10 base hyper-parameters)
    SEED_INSAMPLE = 101
    SEED_OOS = 999
    N_BASE = 4
    N_OOS = 100

    print(f"\nGenerating scenarios (lo=0.8, hi=1.2, mean=1.0, kappa=2.0 | in-sample N={N_BASE}, seed={SEED_INSAMPLE} | out-of-sample N={N_OOS}, seed={SEED_OOS})...")
    scen_det = expected_scenario(members_base)
    scen_in_full = generate_joint_scenarios(members_base, n_joint=10, seed=SEED_INSAMPLE, lo=0.8, hi=1.2, mean=1.0, kappa=2.0)
    scen_in_base = [dict(s) for s in scen_in_full[:N_BASE]]
    for sc in scen_in_base:
        sc['prob'] = 1.0 / N_BASE
    scen_oos = generate_joint_scenarios(members_base, n_joint=N_OOS, seed=SEED_OOS, lo=0.8, hi=1.2, mean=1.0, kappa=2.0)

    print("Computing Individual Contract (I) benchmarks per scenario...")
    ir_det = [compute_individual_benchmarks(com, scen_det[0]['members'])]
    ir_in_base = [compute_individual_benchmarks(com, sc['members']) for sc in scen_in_base]
    ir_oos = [compute_individual_benchmarks(com, sc['members']) for sc in scen_oos]

    # 2. Solve Deterministic Baseline & Three Stochastic Variants on N_BASE in-sample scenarios
    models_cfg = [
        ("DP_det (Deterministic)", scen_det, ir_det, dict(bb_mode='expectation', ir_mode='expectation', risk_weight=0.0, cap_mode='penalty')),
        ("SDP_RN (Risk-Neutral)", scen_in_base, ir_in_base, dict(bb_mode='expectation', ir_mode='none', risk_weight=0.0, cap_mode='penalty')),
        ("SDP_CVaR (Tail Hedge)", scen_in_base, ir_in_base, dict(bb_mode='expectation', ir_mode='none', risk_weight=0.5, cvar_alpha=0.80, cap_mode='penalty')),
        ("SDP_CC (Chance-Constr.)", scen_in_base, ir_in_base, dict(bb_mode='expectation', ir_mode='none', risk_weight=0.0, cap_mode='chance', eps_cap=0.20)),
    ]

    oos_results = {}
    print("\n" + "=" * 128)
    print(f"{'Model':<24} | {'Time(s)':<7} | {'In-Obj':<8} | {'OOS-Mean':<8} | {'OOS-Q90':<8} | {'OOS-CVaR90':<10} | {'CapViol%':<8} | {'AvgExc(kWh)':<11} | {'BB-Def':<7} | {'IR-Viol%':<8}")
    print("=" * 128)

    for label, sc_train, ir_train, kwargs in models_cfg:
        sdp = StochasticDynamicPricingModel(sc_train, contract_data, day_data, ir_bounds_per_scen=ir_train, **kwargs)
        sdp.build_model()
        prices, obj_in, solve_t, st = sdp.solve()
        if prices is None:
            status_txt = "Infeasible" if st == GRB.INFEASIBLE else f"NoSol(status={st})"
            print(f"{label:<24} | {status_txt:<100}")
            continue

        ev_oos = evaluate_prices_on_scenarios(prices, scen_oos, contract_data, day_data, ir_oos)
        oos_results[label] = ev_oos

        print(f"{label:<24} | {solve_t:<7.2f} | {obj_in:<8.2f} | {ev_oos['mean_cost']:<8.2f} | "
              f"{ev_oos['q90_cost']:<8.2f} | {ev_oos['cvar90_cost']:<10.2f} | {ev_oos['cap_viol_prob']:>6.1f}% | "
              f"{ev_oos['mean_exc_kwh']:<11.2f} | {ev_oos['mean_bb_deficit']:<7.2f} | {ev_oos['ir_viol_prob']:>6.1f}%")
    print("=" * 128)

    # 3. Study 1: Accuracy vs Performance as In-Sample Scenario Count N Grows (N = 1, 3, 5, 8, 10)
    print("\n=== Study 1: In-Sample vs Out-of-Sample Convergence across Scenario Count N (SDP_RN) ===")
    print(f"{'N (Scenarios)':<14} | {'In-Sample Obj':<14} | {'OOS Mean':<10} | {'Optimism Gap':<13} | {'OOS CVaR90':<11} | {'CapViol%':<9} | {'Solve Time(s)':<13}")
    print("-" * 98)

    n_sweep = [1, 2, 3, 4]
    conv_records = []
    for n_val in n_sweep:
        sc_sub = [dict(s) for s in scen_in_full[:n_val]]
        for s_item in sc_sub:
            s_item['prob'] = 1.0 / n_val

        sdp_n = StochasticDynamicPricingModel(sc_sub, contract_data, day_data, bb_mode='expectation', ir_mode='none', risk_weight=0.0, time_limit=60)
        sdp_n.build_model()
        pr_n, obj_in_n, t_n, _ = sdp_n.solve()
        if pr_n is not None:
            ev_oos_n = evaluate_prices_on_scenarios(pr_n, scen_oos, contract_data, day_data, ir_oos)
            gap = ev_oos_n['mean_cost'] - obj_in_n
            print(f"{n_val:<14} | {obj_in_n:<14.2f} | {ev_oos_n['mean_cost']:<10.2f} | {gap:<13.2f} | "
                  f"{ev_oos_n['cvar90_cost']:<11.2f} | {ev_oos_n['cap_viol_prob']:>7.1f}% | {t_n:<13.2f}")
            conv_records.append((n_val, obj_in_n, ev_oos_n['mean_cost'], t_n))

    # 4. Study 2: Sensitivity across Risk-Aversion Parameter lambda in SDP_CVaR
    print(f"\n=== Study 2: Sensitivity on Risk-Aversion Weight lambda (SDP_CVaR, N={N_BASE}) ===")
    print(f"{'lambda':<8} | {'OOS Mean (DKK)':<15} | {'OOS CVaR90 (DKK)':<17} | {'OOS Max (DKK)':<14} | {'CapViol%':<9} | {'AvgExc (kWh)':<12}")
    print("-" * 85)

    lambdas = [0.0, 0.25, 0.50, 0.75, 1.0]
    frontier_records = []
    for lam in lambdas:
        sdp_l = StochasticDynamicPricingModel(
            scen_in_base,
            contract_data,
            day_data,
            bb_mode='expectation',
            ir_mode='none',
            risk_weight=lam,
            cvar_alpha=0.80,
            time_limit=90,
        )
        sdp_l.build_model()
        pr_l, _, _, st_l = sdp_l.solve()
        if pr_l is not None:
            ev_oos_l = evaluate_prices_on_scenarios(pr_l, scen_oos, contract_data, day_data, ir_oos)
            print(
                f"{lam:<8.2f} | {ev_oos_l['mean_cost']:<15.2f} | {ev_oos_l['cvar90_cost']:<17.2f} | "
                f"{ev_oos_l['max_cost']:<14.2f} | {ev_oos_l['cap_viol_prob']:>7.1f}% | {ev_oos_l['mean_exc_kwh']:<12.2f}"
            )
            frontier_records.append((lam, ev_oos_l['mean_cost'], ev_oos_l['cvar90_cost'], ev_oos_l['mean_exc_kwh']))
        else:
            status_l = "Infeasible" if st_l == GRB.INFEASIBLE else f"NoSol(status={st_l})"
            print(f"{lam:<8.2f} | {status_l:<70}")

    # 5. Generate Table-13 Compliant Figures in figures/
    fig_dir = (project_root / "figures") if project_root is not None else Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure Q3-1: Empirical CDF of Realized Out-of-Sample Costs (Distribution Plot)
    if oos_results:
        plt.figure(figsize=(9, 5))
        colors = {'DP_det (Deterministic)': 'black', 'SDP_RN (Risk-Neutral)': 'blue', 'SDP_CVaR (Tail Hedge)': 'red', 'SDP_CC (Chance-Constr.)': 'green'}
        linestyles = {'DP_det (Deterministic)': '--', 'SDP_RN (Risk-Neutral)': '-', 'SDP_CVaR (Tail Hedge)': '-.', 'SDP_CC (Chance-Constr.)': ':'}
        for label, res in oos_results.items():
            sorted_c = np.sort(res['costs'])
            cdf = np.arange(1, len(sorted_c) + 1) / len(sorted_c)
            plt.step(sorted_c, cdf, where='post', label=f"{label} (Mean={res['mean_cost']:.1f}, CVaR90={res['cvar90_cost']:.1f})",
                     color=colors.get(label, 'gray'), linestyle=linestyles.get(label, '-'), linewidth=2)
        plt.xlabel('Realized Out-of-Sample Community Cost (DKK)', fontsize=11)
        plt.ylabel('Empirical Cumulative Probability', fontsize=11)
        plt.title('Question 3: Out-of-Sample Realized Cost Distribution (100 Joint Scenarios)', fontsize=12, fontweight='bold')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(fig_dir / 'q3_oos_cost_distribution.png', dpi=300)

    # Figure Q3-2: Convergence of In-Sample vs Out-of-Sample Cost and Solve Time over N
    if conv_records:
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ns = [r[0] for r in conv_records]
        in_m = [r[1] for r in conv_records]
        oos_m = [r[2] for r in conv_records]
        t_m = [r[3] for r in conv_records]

        ax1.plot(ns, in_m, 'b--o', linewidth=2, label='In-Sample Objective (DKK)')
        ax1.plot(ns, oos_m, 'b-s', linewidth=2, label='Out-of-Sample Mean Cost (DKK)')
        ax1.set_xlabel('Number of In-Sample Joint Scenarios (N)', fontsize=11)
        ax1.set_ylabel('Community Cost (DKK)', color='b', fontsize=11)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, linestyle=':', alpha=0.6)

        ax2 = ax1.twinx()
        ax2.plot(ns, t_m, 'r-.^', linewidth=2, label='MILP Solution Time (s)')
        ax2.set_ylabel('Solution Time (s)', color='r', fontsize=11)
        ax2.tick_params(axis='y', labelcolor='r')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
        plt.title('Question 3: In-Sample vs Out-of-Sample Convergence & Computational Time', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(fig_dir / 'q3_scenario_convergence.png', dpi=300)

    # Figure Q3-3: Risk-Aversion Trade-Off Frontier (Mean Cost vs CVaR90)
    if frontier_records:
        plt.figure(figsize=(8, 5))
        f_means = [r[1] for r in frontier_records]
        f_cvars = [r[2] for r in frontier_records]
        plt.plot(f_means, f_cvars, 'purple', marker='o', linewidth=2)
        for lam, mc, cv, _ in frontier_records:
            plt.annotate(rf'$\lambda={lam:.2f}$', (mc, cv), textcoords="offset points", xytext=(6, 4), fontsize=9)
        plt.xlabel('Out-of-Sample Mean Cost (DKK)', fontsize=11)
        plt.ylabel('Out-of-Sample CVaR 90% Cost (DKK)', fontsize=11)
        plt.title('Question 3: Expected Cost vs Tail-Risk Trade-Off Frontier', fontsize=12, fontweight='bold')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(fig_dir / 'q3_risk_tradeoff_frontier.png', dpi=300)

    print(f"\n[Success] Question 3 complete. Saved 3 figures to '{fig_dir}'.")


if __name__ == "__main__":
    main()
