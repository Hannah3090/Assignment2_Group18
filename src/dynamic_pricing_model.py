import gurobipy as gp
from gurobipy import GRB

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

