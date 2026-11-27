from src.dynamic_pricing_model import DynamicPricingModel 
from src.q2_experiments import compute_q1_member_costs, solve_centralized_scaled, run_q2_base_case

def run_q2_cap_sensitivity(com, members_data, contract_data, day_data):
    
    print("\n=== Sensitivity Study: Tightening DSO Import Cap (cap_scale) ===")
    print(f"{'Cap Scale':<10} | {'Cap (kW)':<9} | {'Model I':<9} | {'Central C':<9} | {'DP_cap':<9} | {'DP_BB':<9} | {'DP_IR':<9} | {'DP_BB+IR':<11} | {'Blocking Members (if infeasible)'}")
    print("-" * 125)

    cap_scales = [1.0, 0.8, 0.6, 0.4]
    sens_records = []
    cap_val_default = contract_data["cap"][0]

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
        
    return sens_records