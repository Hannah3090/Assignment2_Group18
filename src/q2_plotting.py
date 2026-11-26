import matplotlib.pyplot as plt

from pathlib import Path

fig_dir = Path("figures")
fig_dir.mkdir(exist_ok=True)

hours = list(range(24))
 
# Figure 1: Prices Comparison
def plot_q2_prices(results_com, prices_cap, prices_bb, prices_ir, prices_bb_ir, day_data):
    plt.figure(figsize=(10, 5))
    if 'dual_power_balance' in results_com:
        plt.step(hours, results_com['lambda_t'], where='mid',
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
    #print(f"\n[Success] Saved all figures to figures/.")
    plt.savefig('dp_prices_comparison.png', dpi=300)

# Figure 2: Net Import Profile
def plot_q2_net_profile(res_cap, res_bb, res_ir, res_bb_ir, contract_data):

    cap_profile = contract_data["cap"]

    plt.figure(figsize=(10, 5))
    if res_cap:
        plt.plot(hours, res_cap['net_import'], label=r'$\mathcal{DP}^{cap}$ (Cap Only)', color='blue', linewidth=2, marker='o')
    if res_bb:
        plt.plot(hours, res_bb['net_import'], label=r'$\mathcal{DP}^{BB}$ (Budget Balance)', color='green', linewidth=2, marker='s')
    if res_ir:
        plt.plot(hours, res_ir['net_import'], label=r'$\mathcal{DP}^{IR}$ (Indiv. Rationality)', color='red', linewidth=2, marker='^')
    if res_bb_ir:
        plt.plot(hours, res_bb_ir['net_import'], label=r'$\mathcal{DP}^{BB+IR}$ (Combined)', color='purple', linewidth=2, linestyle='-.', marker='d')

    # plt.plot(hours, contract_data['cap'],label=rf'DSO Import Cap ($\overline{{P}}_t^{{DSO}} = {cap_val_default:.2f}$ kW)', color='black', linestyle='--', alpha=0.8)

    plt.plot(hours, contract_data['cap'], label="DSO Import Cap", color='black', linestyle='--', alpha=0.8)
    plt.axhline(0, color='gray', linestyle=':', linewidth=1.0)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Net Power Import (kW)', fontsize=12)
    plt.title('Question 2: Aggregated Community Net Power Profile (Base Case)', fontsize=13, fontweight='bold')
    plt.xticks(hours)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    print(f"\n[Success] Saved all figures to figures/.")
    plt.savefig('community_net_profile.png', dpi=300)

    # Figure 3: Sensitivity Study Plot
def plot_q2_cap_sensitivity(sens_records):
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
    print(f"\n[Success] Saved all figures to figures/.")
    plt.savefig('dp_sensitivity_cap.png', dpi=300)



