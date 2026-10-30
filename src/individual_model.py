import gurobipy as gp
import pandas as pd
from gurobipy import GRB

from member_model import MemberModel

class IndividualModel:

    def __init__(self, community):
        self.community = community

    def solve(self,return_members=False):

        results = {}

        for member in self.community.members:

            model = MemberModel(member, mode="I", contract=self.community.contract, consumer_id=member.consumer_id)

            summary, hourly = model.solve()

            results[member.consumer_id] = {
                "summary": summary,
                "hourly": hourly
            }

        # this "return" is created for settlement analysis' comparison!
        if return_members:
            return {
                    "aggregate": self.aggregate(results),
                    "member_results": results
                    }

        return self.aggregate(results)

    def aggregate(self, member_results):

        T = range(24)

        community_import = [0.0] * 24
        community_export = [0.0] * 24

        for cid, res in member_results.items():

            hourly = res["hourly"]

            for t in T:

                community_import[t] += hourly["p_im"].iloc[t]
                community_export[t] += hourly["p_ex"].iloc[t]

        # RESULTS FOR REPORTING:
        daily_import = sum(community_import)

        peak_import = max(community_import)

        cap = self.community.contract.capacity_limit_kW

        exceedance = sum(
            max(0.0, community_import[t] - cap[t])
            for t in T
        )

        total_cost = 0

        for cid, res in member_results.items():
            total_cost += res["summary"]["objective (2.1) = C + D_L [DKK]"]

        return {
            "member_results": member_results,
            "community_import": community_import,
            "community_export": community_export,
            "daily_import": daily_import,
            "peak_import": peak_import,
            "exceedance": exceedance,
            "total_cost": total_cost
        }

