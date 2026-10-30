"""     def __init__(self, data):
        self.data = data
        self.T = range(1, data.n_hours + 1)                  # hours t = 1, ..., 24
        self.m = gp.Model("Q2b_linear_disutility_consumer")
        self.m.Params.OutputFlag = 0
        self.build() 
"""
import gurobipy as gp
import pandas as pd
from gurobipy import GRB

class MemberModel:

    def __init__(self, data, mode="R", contract=None, consumer_id=None):

        self.data = data

        self.mode = mode
        self.contract = contract
        self.consumer_id = consumer_id

        self.T = range(1, data.n_hours + 1)

        self.m = gp.Model(f"Member_{mode}")
        self.m.Params.OutputFlag = 0

        self.build()

    def build(self):
        d, m, T = self.data, self.m, self.T

        t_end = T[-1]                                        # last hour, 24

        # Input parameters (hourly ones as dicts indexed by t = 1..24)
        pi_DA = {t: d.energy_price[t - 1] for t in T}
        PV_bar = {t: d.pv_available[t - 1] for t in T}
        l_ref = {t: d.reference_load[t - 1] for t in T}        # reference consumption, kWh
        c_PV, c_L = d.pv_marginal_cost, d.linear_disutility    # DKK/kWh
        L_min, L_max = d.load_lower_kWh, d.load_upper_kWh
        E_min = d.min_daily_energy_kWh

        #------------------- TARIFF STRUCTURE -----------------------

        beta = 0.0

        if self.mode == "I":
            beta = self.contract.tariff_discount_ratio

        pi_im = {t: pi_DA[t] + (1-beta) * d.import_tariff for t in T}

        pi_ex = {t: pi_DA[t] - (1-beta) * d.export_tariff for t in T}

        # Decision variables (kWh in hour t): all unbounded, every bound is an explicit constraint
        self.p_im = m.addVars(T, lb=-GRB.INFINITY, name="p_im")
        self.p_ex = m.addVars(T, lb=-GRB.INFINITY, name="p_ex")
        self.p_PV = m.addVars(T, lb=-GRB.INFINITY, name="p_PV")
        self.l = m.addVars(T, lb=-GRB.INFINITY, name="l")
        self.d_pos = m.addVars(T, lb=-GRB.INFINITY, name="d_pos")   # consumption above the reference
        self.d_neg = m.addVars(T, lb=-GRB.INFINITY, name="d_neg")   # consumption below the reference

        p_im, p_ex, p_PV, l, d_pos, d_neg = self.p_im, self.p_ex, self.p_PV, self.l, self.d_pos, self.d_neg

        # FOR INDIVIDUAL CONTRACTS, EXCEEDANCE VARIABLE CREATED.
        if self.mode == "I":
            self.e = m.addVars(T, lb=0, name="exceedance")

        self.has_battery = (d.battery_capacity_kWh is not None)

        if self.has_battery:

            S_bar = d.battery_capacity_kWh
            S_init = d.battery_initial_soc_kWh

            P_ch_bar = d.battery_max_charge_kW
            P_dis_bar = d.battery_max_discharge_kW

            eta_ch = d.battery_charging_efficiency
            eta_dis = d.battery_discharging_efficiency

            self.p_ch = m.addVars(T, lb=-GRB.INFINITY, name="p_ch")     # energy drawn to charge the battery
            self.p_dis = m.addVars(T, lb=-GRB.INFINITY, name="p_dis")   # energy delivered by discharging
            self.s = m.addVars(T, lb=-GRB.INFINITY, name="s")           # state of charge at the end of hour t

            p_ch, p_dis, s = self.p_ch, self.p_dis, self.s

        # Objective (2.1): procurement cost C plus linear disutility D_L
        self.C = gp.quicksum(pi_im[t] * p_im[t] - pi_ex[t] * p_ex[t] + c_PV * p_PV[t] for t in T)
        self.D = gp.quicksum(c_L * (d_pos[t] + d_neg[t]) for t in T)

        objective = self.C + self.D

        if self.mode == "I":

            penalty = self.contract.penalty_DKK_per_kWh  #PENALTY term is added for individual contracts case.

            self.P = gp.quicksum(penalty * self.e[t] for t in T)

            objective += self.P

        m.setObjective(objective, GRB.MINIMIZE)

        # Constraints, written exactly as h(x) = 0 or g(x) <= 0 in the solution text
        c = self.con = {}

        c["PV_up"] = m.addConstrs((p_PV[t] - PV_bar[t] <= 0 for t in T), name="PV_up")      
        c["PV_lo"] = m.addConstrs((-p_PV[t] <= 0 for t in T), name="PV_lo")                       

        c["l_up"] = m.addConstrs((l[t] <= L_max[t-1] for t in T), name="l_up")
        c["l_lo"] = m.addConstrs((l[t] >= L_min[t-1] for t in T), name="l_lo")

        c["im_lo"] = m.addConstrs((-p_im[t] <= 0 for t in T), name="im_lo")                       
        c["ex_lo"] = m.addConstrs((-p_ex[t] <= 0 for t in T), name="ex_lo")                      

        c["split"] = m.addConstrs(
            (l[t] - l_ref[t] - d_pos[t] + d_neg[t] == 0 for t in T), name="split")            
        c["dpos_lo"] = m.addConstrs((-d_pos[t] <= 0 for t in T), name="dpos_lo")                  
        c["dneg_lo"] = m.addConstrs((-d_neg[t] <= 0 for t in T), name="dneg_lo")                  

        c["E_min"] = m.addConstr(gp.quicksum(l[t] for t in T) >= d.min_daily_energy_kWh, name="E_min")

        if self.has_battery:

            c["bal"] = m.addConstrs((l[t] + p_ex[t] + self.p_ch[t] - p_im[t] - p_PV[t] - self.p_dis[t] == 0 for t in T), name="bal")

            c["soc_1"] = m.addConstr(s[1] - S_init - eta_ch * p_ch[1] + p_dis[1] / eta_dis == 0, name="soc_1")              
            c["soc"] = m.addConstrs((s[t] - s[t - 1] - eta_ch * p_ch[t] + p_dis[t] / eta_dis == 0 for t in T if t > 1), name="soc")   

            c["ch_up"] = m.addConstrs((p_ch[t] - P_ch_bar <= 0 for t in T), name="ch_up")            
            c["dis_up"] = m.addConstrs((p_dis[t] - P_dis_bar <= 0 for t in T), name="dis_up")      
            c["s_up"] = m.addConstrs((s[t] - S_bar <= 0 for t in T), name="s_up")                   

            c["ch_lo"] = m.addConstrs((-p_ch[t] <= 0 for t in T), name="ch_lo")                       
            c["dis_lo"] = m.addConstrs((-p_dis[t] <= 0 for t in T), name="dis_lo")                   

            c["s_lo"] = m.addConstrs((-s[t] <= 0 for t in T), name="s_lo")                 
            c["s_end"] = m.addConstr(S_init - s[t_end] <= 0, name="s_end")                

        else:
            c["bal"] = m.addConstrs((l[t] + p_ex[t] - p_im[t] - p_PV[t] == 0 for t in T), name="bal")

        # CAP CONSTRATINT for individual contracts:

        if self.mode == "I":
        
            cap_profile = self.contract.individual_capacity_limit_kW[self.consumer_id]
            c["cap"] = m.addConstrs((p_im[t] - self.e[t] <= cap_profile[t-1] for t in T), name="cap")  

    def solve(self):
        """Solve; return (summary dict, hourly DataFrame). KKT multiplier = -Pi (see consumer_q1.py)."""

        d, m, T, c = self.data, self.m, self.T, self.con
        m.optimize()

        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"{m.ModelName}: Gurobi status {m.Status}")
        X = lambda var: [var[t].X for t in T]
        Y = lambda con: [-con[t].Pi for t in T]

        hourly = pd.DataFrame({
            "pi_DA": d.energy_price, "PV_bar": d.pv_available, "l_ref": d.reference_load,
            "l": X(self.l), "d_pos": X(self.d_pos), "d_neg": X(self.d_neg),
            "p_PV": X(self.p_PV), "p_im": X(self.p_im), "p_ex": X(self.p_ex),
            "p_ch": X(self.p_ch) if self.has_battery else [None]*len(T), "p_dis": X(self.p_dis) if self.has_battery else None, "s": X(self.s) if self.has_battery else None,
            "lambda": Y(c["bal"]), "rho_up": Y(c["PV_up"]), "rho_lo": Y(c["PV_lo"]),
            "omega_up": Y(c["l_up"]), "omega_lo": Y(c["l_lo"]), "alpha": Y(c["im_lo"]), "beta": Y(c["ex_lo"]),
            "theta": Y(c["split"]), "psi_pos": Y(c["dpos_lo"]), "psi_neg": Y(c["dneg_lo"]),
        }, index=pd.Index(T, name="t"))

        summary = {
            "objective (2.1) = C + D_L [DKK]": m.ObjVal,
            "procurement cost C [DKK]": self.C.getValue(),
            "disutility D_L [DKK]": self.D.getValue(),
            "mu (E_min)": -c["E_min"].Pi
        }

        #Penalty value is extracted for individual contracts:
        if self.mode == "I":
            summary["penalty cost [DKK]"] = self.P.getValue()

        return summary, hourly
