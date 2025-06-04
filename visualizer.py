import os
from datetime import datetime
import pandas as pd


def analyze_step1_results(
    soc_daa_hours: list,
    cha_daa_quarters: list,
    dis_daa_quarters: list,
    price_list_hourly: list,
    power_cap: float,
    n_hours: int,
):
    """Basic printout of the optimization results of step 1."""
    df = pd.DataFrame({
        "Hour": range(1, len(soc_daa_hours) + 1),
        "SoC_kWh": soc_daa_hours,
        "Price_DAA_EUR_per_kWh": price_list_hourly[: len(soc_daa_hours)],
    })
    print("Step1 results preview:\n", df.head())
    print(
        f"Charged quarters: {sum(cha_daa_quarters):.2f}, discharged quarters: {sum(dis_daa_quarters):.2f}"
    )


def calculate_real_cycles(
    cha_quarters: list,
    dis_quarters: list,
    power_cap: float,
    n_hours: int,
    min_soc: float,
    max_soc: float,
    allowed_cycles: int,
    eta_cha: float,
    eta_dis: float,
    **kwargs,
):
    """Rough estimation of performed battery cycles."""
    energy_charged = sum(cha_quarters) * power_cap / 4 * eta_cha
    energy_discharged = sum(dis_quarters) * power_cap / 4 / max(eta_dis, 1e-9)
    throughput = (energy_charged + energy_discharged) / 2
    full_cycle_energy = max(max_soc - min_soc, 1e-9)
    cycles = throughput / full_cycle_energy
    print(f"Estimated real cycles: {cycles:.2f} of allowed {allowed_cycles}")
    return cycles


def combine_charge_discharge(cha_quarters: list, dis_quarters: list, power_cap: float):
    """Combine charge/discharge to an energy profile in kWh per quarter."""
    return [
        (dis - cha) * power_cap / 4
        for cha, dis in zip(cha_quarters, dis_quarters)
    ]


def export_full_results_to_excel_premium(
    soc_hours: list,
    cha_quarters: list,
    dis_quarters: list,
    energy_profile: list,
    profit: float,
    n_days: int,
    power_cap: float,
    energy_cap: float,
    min_soc: float,
    max_soc: float,
    eta_cha: float,
    eta_dis: float,
    n_cycles: int,
    cha_daa_h: list,
    dis_daa_h: list,
    price_list_daa: list,
    c_rate: float,
    price_list_ida: list,
    folder_path: str = ".",
):
    """Export a summary of the optimization into an Excel file."""
    os.makedirs(folder_path, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder_path, f"results_{ts}.xlsx")

    df = pd.DataFrame({
        "Hour": range(1, len(soc_hours) + 1),
        "SoC_kWh": soc_hours,
        "Charge_pu": cha_daa_h,
        "Discharge_pu": dis_daa_h,
        "Price_DAA": price_list_daa[: len(soc_hours)],
    })
    df.to_excel(filename, index=False)
    print(f"Results exported to {filename}")


def auswertung_transaktionen_stuendlich(
    cha_daa_h: list,
    dis_daa_h: list,
    price_list_daa: list,
    power_cap: float,
    eta_dis: float,
):
    """Simple hourly transaction summary."""
    revenue = 0.0
    for i, (c, d) in enumerate(zip(cha_daa_h, dis_daa_h)):
        net = (d - c) * power_cap
        revenue += net * price_list_daa[i]
    print(f"Estimated revenue step1: {revenue:.2f} EUR")
    return revenue
