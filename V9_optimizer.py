# 📜 BESCHREIBUNG:
# Dieses Python-Skript definiert eine Optimierungsklasse für die Betriebsführung eines Batteriespeichers.
# Es wird ein Lade-/Entladeprofil für den Day-Ahead-Markt (DAA) und den Intraday-Continuous-Markt (IDC) berechnet,
# um den Gewinn durch Stromhandel zu maximieren.
# 
# Die Optimierung berücksichtigt:
# - Lade- und Entladeverluste (Wirkungsgrade)
# - Maximale und minimale Ladezustände (State of Charge, SoC)
# - Begrenzung der Ladezyklen
# - Viertelstündliche Auflösung im IDC-Markt
#
# Besonderheit dieser Version:
# → Laden und Entladen sind **stufenlos möglich** (zwischen 0% und 100% Ladeleistung).
# → Gleichzeitig Laden und Entladen wird **physikalisch ausgeschlossen** durch Binärvariablen.
# → Dadurch kann das Modell intelligent und realitätsnah entscheiden, wann welche Energiemengen optimal gehandelt werden.
# → Im IDC wird auf Basis des Ergebnisses aus DAA optimiert, indem viertelstündliche Anpassungen vorgenommen werden.

import pyomo.environ as pyo
import numpy as np
import pandas as pd





class optimizer:
    @staticmethod
    def _load_price_list(path: str) -> list:
        """Load a price column from the Excel sheet and return values in EUR/kWh."""
        df = pd.read_excel(path, sheet_name="1")
        series = df.iloc[24:, 1]
        clean = series.astype(str).str.replace(",", ".")
        return clean.astype(float).div(1000).tolist()
    def __init__(
        self,
        n_days: int,
        n_cycles: int,
        c_rate: float,
        energy_cap: float,
        eta_cha: float,
        eta_dis: float,
        min_soc: float,
        max_soc: float,
        excel_path_ida,
        excel_path_daa,
    ):
            # Batterieparameter
            self.n_days = n_days
            self.n_hours = int(24 * n_days)
            self.n_cycles = n_cycles
            self.c_rate = c_rate
            self.energy_cap = energy_cap  # kWh
            self.power_cap = energy_cap * c_rate  # kW
            self.eta_cha = eta_cha
            self.eta_dis = eta_dis
            self.min_soc = min_soc
            self.max_soc = max_soc



            # Preislisten
            self.price_list_daa = self._load_price_list(excel_path_daa)
            self.price_list_ida = self._load_price_list(excel_path_ida)
            self.price_list_daa_q = np.repeat(self.price_list_daa, 4)





    def set_highs_solver(self):
        return pyo.SolverFactory("highs")
    
        
    


    def step1_optimize_daa(self):
        model = pyo.ConcreteModel()
        n_hours = self.n_hours
        power_cap = self.power_cap
        eta_cha = self.eta_cha
        eta_dis = self.eta_dis
        min_soc = self.min_soc
        max_soc = self.max_soc

        #print(self.price_list_daa[:5])
              

        # 1. Profit-Trigger: Berechne zukünftigen Maximalpreis und Profit-Fähigkeit
        best_future_price = np.maximum.accumulate(self.price_list_daa[::-1])[::-1]
        can_charge = [
            1 if fut * eta_dis > price / eta_cha else 0
            for fut, price in zip(best_future_price, self.price_list_daa)
        ]

        # Indexmengen
        model.T = pyo.RangeSet(1, n_hours)
        model.T_plus_1 = pyo.RangeSet(1, n_hours + 1)

        # Entscheidungsvariablen
        model.soc = pyo.Var(model.T_plus_1, domain=pyo.Reals)
        model.cha_daa = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, 1))
        model.dis_daa = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, 1))
        model.charge_flag = pyo.Var(model.T, domain=pyo.Binary)
        model.discharge_flag = pyo.Var(model.T, domain=pyo.Binary)

        # Param für Profit-Trigger
        model.can_charge = pyo.Param(
            model.T,
            initialize={t+1: can_charge[t] for t in range(n_hours)},
            within=pyo.Binary,
        )

        # Logische Kopplungen: Flags
        model.cha_flag_constraint = pyo.Constraint(model.T, rule=lambda m, t: m.cha_daa[t] <= m.charge_flag[t])
        model.dis_flag_constraint = pyo.Constraint(model.T, rule=lambda m, t: m.dis_daa[t] <= m.discharge_flag[t])
        model.no_simultaneous = pyo.Constraint(model.T, rule=lambda m, t: m.charge_flag[t] + m.discharge_flag[t] <= 1)

        # Profit-Trigger: Nur laden, wenn profitabel
        model.charge_profit_trigger = pyo.Constraint(
            model.T,
            rule=lambda m, t: m.cha_daa[t] <= m.can_charge[t]
        )

        # SoC-Grenzen und Bilanz
        model.set_maximum_soc = pyo.Constraint(model.T_plus_1, rule=lambda m, t: m.soc[t] <= max_soc)
        model.set_minimum_soc = pyo.Constraint(model.T_plus_1, rule=lambda m, t: m.soc[t] >= min_soc)
        model.set_first_soc_to_min = pyo.Constraint(rule=lambda m: m.soc[1] == min_soc)
        model.set_last_soc_to_min = pyo.Constraint(rule=lambda m: m.soc[n_hours + 1] == min_soc)

        # Zyklus-Volumen-Limit
        volume_limit = (self.max_soc - self.min_soc) * self.n_cycles * (n_hours/24)
        model.charge_cycle_limit = pyo.Constraint(
            rule=lambda m: sum(m.cha_daa[t] * power_cap * eta_cha for t in m.T) <= volume_limit
        )
        model.discharge_cycle_limit = pyo.Constraint(
            rule=lambda m: sum(m.dis_daa[t] * power_cap for t in m.T) <= volume_limit
        )





        # SoC-Dynamik
        model.soc_step_constraint = pyo.Constraint(
            model.T,
            rule=lambda m, t: m.soc[t + 1] ==
                m.soc[t]
                + eta_cha * power_cap * m.cha_daa[t]
                - (1 / eta_dis) * power_cap * m.dis_daa[t]
        )

        # Zielfunktion: Profit aus Dis-/Charge
        model.obj = pyo.Objective(
            expr=sum(
                power_cap * self.price_list_daa[t-1] * (model.dis_daa[t] - model.cha_daa[t])
                for t in model.T
            ),
            sense=pyo.maximize
        )

        # Solve
        solver = self.set_highs_solver()
        solver.solve(model, tee=True)

        # Ergebnisse extrahieren
        soc = [model.soc[t].value for t in range(1, n_hours + 1)]
        cha = [model.cha_daa[t].value for t in range(1, n_hours + 1)]
        dis = [model.dis_daa[t].value for t in range(1, n_hours + 1)]

        profit = sum(
            power_cap * price * (d - c)
            for c, d, price in zip(cha, dis, self.price_list_daa)
        )


        # Auf Viertelstunden skalieren
        chaq = np.repeat(cha, 4)
        disq = np.repeat(dis, 4)
        socq = np.repeat(soc, 4)
        cha_real = [float(x * power_cap/4 * eta_cha) for x in chaq]
        dis_real = [float(x * power_cap/4 * (1 / eta_dis)) for x in disq]

        return soc, socq, chaq, disq, profit, cha, dis, cha_real, dis_real



    def step2_optimize_ida(self, step1_cha_daa: list, step1_dis_daa: list):
        model = pyo.ConcreteModel()
        n_hours = self.n_hours
        energy_cap = self.energy_cap
        power_cap = self.power_cap
        eta_cha = self.eta_cha
        eta_dis = self.eta_dis
        min_soc = self.min_soc
        max_soc = self.max_soc
        # Anzahl Viertelstunden
        N = 4 * n_hours

        # 1) Indexmengen
        model.Q        = pyo.RangeSet(1, N)
        model.Q_plus_1 = pyo.RangeSet(1, N + 1)

        # 2) Profit-Flags für Closing-Deals
        can_close_buy  = [1 if self.price_list_ida[i]  < self.price_list_daa_q[i] else 0 for i in range(N)]
        can_close_sell = [1 if self.price_list_ida[i]  > self.price_list_daa_q[i] else 0 for i in range(N)]
        model.can_close_buy  = pyo.Param(model.Q,
            initialize={i+1: can_close_buy[i] for i in range(N)}, within=pyo.Binary)
        model.can_close_sell = pyo.Param(model.Q,
            initialize={i+1: can_close_sell[i] for i in range(N)}, within=pyo.Binary)
        

             # 1. Profit-Trigger: Berechne zukünftigen Maximalpreis und Profit-Fähigkeit
        best_future_price = np.maximum.accumulate(self.price_list_ida[::-1])[::-1]
        can_charge = [
            1 if fut * eta_dis > price / eta_cha else 0
            for fut, price in zip(best_future_price, self.price_list_ida)
        ]

        # Param für Profit-Trigger
        model.can_charge = pyo.Param(
            model.Q,
            initialize={q: can_charge[q-1] for q in range(1, N+1)},
            within=pyo.Binary,
        )



        # 3) Variablen
        model.soc           = pyo.Var(model.Q_plus_1, domain=pyo.Reals)
        model.cha_ida       = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0,1))
        model.dis_ida       = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0,1))
        model.cha_ida_close = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0,1))
        model.dis_ida_close = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0,1))

        # Profit-Trigger: Nur laden, wenn profitabel
        model.charge_profit_trigger = pyo.Constraint(
            model.Q,
            rule=lambda m, t: m.cha_ida[t] <= m.can_charge[t]
        )


        # 4) SoC-Grenzen
        model.set_max_soc   = pyo.Constraint(model.Q_plus_1,
                            rule=lambda m, q: m.soc[q] <= max_soc)
        model.set_min_soc   = pyo.Constraint(model.Q_plus_1,
                            rule=lambda m, q: m.soc[q] >= min_soc)
        model.set_soc_start = pyo.Constraint(rule=lambda m: m.soc[1] == min_soc)
        model.set_soc_end   = pyo.Constraint(rule=lambda m: m.soc[N+1] == min_soc)

        # 5) SoC-Dynamik inkl. DA-Positionen & Closing-Deals
        def soc_step(m, q):
            return m.soc[q+1] == m.soc[q] \
                + power_cap/4 * (
                    eta_cha * m.cha_ida[q]
                - (1/eta_dis) * m.dis_ida[q]
                +     m.cha_ida_close[q]
                -     m.dis_ida_close[q]
                + step1_cha_daa[q-1]
                - step1_dis_daa[q-1]
                )
        model.soc_dynamics = pyo.Constraint(model.Q, rule=soc_step)

        # 6) Zyklus-Limits (kWh pro Tag)
        volume_limit = energy_cap * n_hours / 24 * self.n_cycles  # optional: an Tag anpassen
        model.charge_cycle_limit = pyo.Constraint(
            rule=lambda m:
            sum(m.cha_ida[q] for q in m.Q)*power_cap/4
        + sum(step1_cha_daa    )*power_cap/4
        <= volume_limit
        )
        model.discharge_cycle_limit = pyo.Constraint(
            rule=lambda m:
            sum(m.dis_ida[q] for q in m.Q)*power_cap/4
        + sum(step1_dis_daa    )*power_cap/4
        <= volume_limit
        )

        # 7) Closing-Logik nur bei profitabler Differenz
        model.cha_close_logic = pyo.Constraint(model.Q,
            rule=lambda m,q:
            m.cha_ida_close[q] <= step1_dis_daa[q-1] * m.can_close_buy[q]
        )
        model.dis_close_logic = pyo.Constraint(model.Q,
            rule=lambda m,q:
            m.dis_ida_close[q] <= step1_cha_daa[q-1] * m.can_close_sell[q]
        )

        # 8) Rate-Limits (max. 1 p.u. pro Slot inkl. DAA-Position)
        model.charge_rate_limit    = pyo.Constraint(model.Q,
            rule=lambda m,q: m.cha_ida[q]    + step1_cha_daa[q-1] <= 1
        )
        model.discharge_rate_limit = pyo.Constraint(model.Q,
            rule=lambda m,q: m.dis_ida[q]    + step1_dis_daa[q-1] <= 1
        )

   
         # Zielfunktion: kombiniere DAA- und IDA-Trades
        model.obj = pyo.Objective(
            expr = sum(
                # DAA-Profit pro Slot (viertelstündlich)
                self.price_list_daa_q[q-1] * power_cap/4 * (step1_dis_daa[q-1] - step1_cha_daa[q-1]) +
                # IDA-Profit
                self.price_list_ida[q-1] * power_cap/4 * (model.dis_ida[q] + model.dis_ida_close[q] - model.cha_ida[q] - model.cha_ida_close[q])
                for q in model.Q
            ),
            sense = pyo.maximize
        )
        # 10) Solve & Rückgabe
        solver = self.set_highs_solver()
        solver.solve(model, tee=True)

        # Ergebnisse extrahieren
        soc_ida         = [model.soc[q].value for q in range(1, N+1)]
        cha_ida         = [model.cha_ida[q].value for q in model.Q]
        dis_ida         = [model.dis_ida[q].value for q in model.Q]
        cha_ida_close   = [model.cha_ida_close[q].value for q in model.Q]
        dis_ida_close   = [model.dis_ida_close[q].value for q in model.Q]


        profit_ida = sum(
            self.price_list_daa_q[i] * (
                eta_dis * power_cap / 4 * step1_dis_daa[i]
                - power_cap / 4 / eta_cha * step1_cha_daa[i]
            )
            + self.price_list_ida[i] * (
                eta_dis * power_cap / 4 * (dis_ida[i] + dis_ida_close[i])
                - power_cap / 4 / eta_cha * (cha_ida[i] + cha_ida_close[i])
            )
            for i in range(N)
        )

        # Kombinierte Profile (DAA + IDA inkl. Close-Deals)
        cha_combined = np.asarray(step1_cha_daa)   \
                    - np.asarray(dis_ida_close) \
                    + np.asarray(cha_ida)
        dis_combined = np.asarray(step1_dis_daa)   \
                    - np.asarray(cha_ida_close) \
                    + np.asarray(dis_ida)

        return (
            soc_ida, cha_ida, dis_ida,
            cha_ida_close, dis_ida_close,
            profit_ida, cha_combined.tolist(), dis_combined.tolist()
        )

        