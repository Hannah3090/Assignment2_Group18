"""     SIMILIAR TO MEMBER_MODEL.PY BUT AT COMMUNITY SCALE!
"""

import gurobipy as gp
import pandas as pd
from gurobipy import GRB

class CommunityModel:

    def __init__(self, community):

        self.community = community

        self.m = gp.Model("Model_C")

        self.m.Params.OutputFlag = 0

        self.build()
    
    def build(self):
        
        m = self.m
        T = range(1, 25)

        members = self.community.members

        # DEFINING COMMUNITY VARIABLES:
        self.Q_imp = m.addVars(T, lb=0, name="Q_imp")

        self.Q_exp = m.addVars(T, lb=0, name="Q_exp")

        self.e = m.addVars(T, lb=0, name="e")

        # DEFINING MEMBER VARIABLES:
        self.l = {}
        self.p_PV = {}
        self.d_pos = {}
        self.d_neg = {}

        self.p_ch = {}
        self.p_dis = {}
        self.s = {}

        c = self.con = {}

        for member in members:

            cid = member.consumer_id

            self.l[cid] = m.addVars(T, lb=-GRB.INFINITY, name=f"l_{cid}")

            self.p_PV[cid] = m.addVars(T,lb=0, name=f"pv_{cid}")

            self.d_pos[cid] = m.addVars(T, lb=0, name=f"d_pos_{cid}")

            self.d_neg[cid] = m.addVars(T, lb=0, name=f"d_neg_{cid}")

            if member.battery_capacity_kWh is not None:

                self.p_ch[cid] = m.addVars(T, lb=0, name=f"p_ch_{cid}")

                self.p_dis[cid] = m.addVars(T, lb=0, name=f"p_dis_{cid}")

                self.s[cid] = m.addVars(T, lb=0, name=f"s_{cid}")

                S_bar = member.battery_capacity_kWh
                S_init = member.battery_initial_soc_kWh

                P_ch_bar = member.battery_max_charge_kW
                P_dis_bar = member.battery_max_discharge_kW

                eta_ch = member.battery_charging_efficiency
                eta_dis = member.battery_discharging_efficiency

                c[f"soc_1_{cid}"] = m.addConstr(self.s[cid][1] - S_init - eta_ch * self.p_ch[cid][1] + self.p_dis[cid][1] / eta_dis == 0, name=f"soc_1_{cid}")

                c[f"soc_{cid}"] = m.addConstrs((self.s[cid][t] - self.s[cid][t-1] - eta_ch * self.p_ch[cid][t] + self.p_dis[cid][t] / eta_dis == 0 for t in T if t > 1), name=f"soc_{cid}")

                c[f"ch_up_{cid}"] = m.addConstrs((self.p_ch[cid][t] <= P_ch_bar for t in T), name=f"ch_up_{cid}")

                c[f"dis_up_{cid}"] = m.addConstrs((self.p_dis[cid][t] <= P_dis_bar for t in T), name=f"dis_up_{cid}")

                c[f"s_up_{cid}"] = m.addConstrs((self.s[cid][t] <= S_bar for t in T), name=f"s_up_{cid}")

                c[f"s_end_{cid}"] = m.addConstr(self.s[cid][24] >= S_init, name=f"s_end_{cid}")
      
            PV_bar = member.pv_available
            l_ref = member.reference_load

            L_min = member.load_lower_kWh
            L_max = member.load_upper_kWh

            E_min = member.min_daily_energy_kWh

            c[f"PV_up_{cid}"] = m.addConstrs((self.p_PV[cid][t] <= PV_bar[t-1] for t in T), name=f"PV_up_{cid}")

            c[f"l_up_{cid}"] = m.addConstrs((self.l[cid][t] <= L_max[t-1] for t in T), name=f"l_up_{cid}")
            
            c[f"l_lo_{cid}"] = m.addConstrs((self.l[cid][t] >= L_min[t-1] for t in T), name=f"l_lo_{cid}")

            c[f"split_{cid}"] = m.addConstrs((self.l[cid][t] - l_ref[t-1] - self.d_pos[cid][t] + self.d_neg[cid][t] == 0 for t in T), name=f"split_{cid}")
        
            c[f"E_min_{cid}"] = m.addConstr(gp.quicksum(self.l[cid][t] for t in T) >= E_min, name=f"E_min_{cid}")

        c["community_balance"] = m.addConstrs((gp.quicksum(self.l[mem.consumer_id][t] +
                                                                (self.p_ch[mem.consumer_id][t] if mem.consumer_id in self.p_ch else 0) -
                                                                self.p_PV[mem.consumer_id][t] - 
                                                                (self.p_dis[mem.consumer_id][t] if mem.consumer_id in self.p_dis else 0) for mem in members) +
                                                                self.Q_exp[t] - self.Q_imp[t] == 0 for t in T), name="community_balance")
            
        cap = self.community.contract.capacity_limit_kW

        c["community_cap"] = m.addConstrs((self.Q_imp[t] - self.e[t] <= cap[t-1] for t in T), name="community_cap")

        beta = self.community.contract.tariff_discount_ratio

        penalty = self.community.contract.penalty_DKK_per_kWh
       
        self.C = (gp.quicksum((members[0].energy_price[t-1] + (1-beta) * members[0].import_tariff) * self.Q_imp[t] -
                              (members[0].energy_price[t-1] - (1-beta) * members[0].export_tariff) * self.Q_exp[t] for t in T) +
                 gp.quicksum(member.pv_marginal_cost * self.p_PV[member.consumer_id][t] for member in members for t in T))
        
        self.D = gp.quicksum(member.linear_disutility * (self.d_pos[member.consumer_id][t] + self.d_neg[member.consumer_id][t]) for member in members for t in T)
        
        self.P = gp.quicksum(penalty * self.e[t] for t in T)

        m.setObjective(self.C + self.D + self.P, GRB.MINIMIZE)
        
    def solve(self):

        self.m.optimize()

        if self.m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi status {self.m.Status}")
        
        # RESULTS FOR REPORTING:         
        lambda_t = [-self.con["community_balance"][t].Pi for t in range(1,25)]

        community_import = [self.Q_imp[t].X for t in range(1,25)]

        #print("\nCommunity import profile:")
        #print(community_import)

        community_export = [self.Q_exp[t].X for t in range(1,25)]

        #print("Community export:")
        #print(community_export)

        member_net = {}

        for member in self.community.members:

            cid = member.consumer_id

            profile = []

            for t in range(1,25):

                load = self.l[cid][t].X

                pv = self.p_PV[cid][t].X

                p_ch = (self.p_ch[cid][t].X  if cid in self.p_ch else 0)

                p_dis = (self.p_dis[cid][t].X if cid in self.p_dis else 0)

                net = load + p_ch - pv - p_dis

                profile.append(net)

            member_net[cid] = profile

        return {
            "daily_import": sum(community_import),
            "peak_import": max(community_import),
            "exceedance": sum(self.e[t].X for t in range(1,25)),
            "total_cost": self.m.ObjVal,
            "C": self.C.getValue(),
            "D": self.D.getValue(),
            "P": self.P.getValue(),
            "lambda": lambda_t,
            "member_net": member_net
        }
    
