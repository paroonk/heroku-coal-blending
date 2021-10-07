import json
import math
import time
from datetime import datetime
from itertools import product

import django_rq
import gspread
import numpy as np
import pandas as pd
from decouple import config
from django_rq import job
from oauth2client.service_account import ServiceAccountCredentials
from pyomo.environ import *
from pyomo.opt import results
from tqdm import tqdm

pd.options.display.float_format = "{:,.2f}".format
pd.options.display.max_columns = None
pd.options.display.expand_frame_repr = False

input_url = 'https://docs.google.com/spreadsheets/d/1rflUWAGZd0vlKGxEGR_kn7LuVmegOQiL8c35Nzh6lmM/edit#gid=486638920'
result_url = 'https://docs.google.com/spreadsheets/d/1N2EiCGQMSnxOmzuFyE6cnwrLoBTLdrjF-h7l2cUpAo0/edit#gid=979337795'
Start_Date = '2019-12-31'
Sim_Week_Total =  52
lime_price = 690
flyash_price = 250
bottomash_price = 757.95

# use creds to create a google client to interact with the Google Drive API
json_creds = config('GOOGLE_APPLICATION_CREDENTIALS')
creds_dict = json.loads(json_creds)
creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
scopes = ['https://spreadsheets.google.com/feeds'] 
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scopes)
client = gspread.authorize(creds)

@job('default', timeout=10800)
def coal_optimize():
    # _______Import______

    # Find a workbook by url
    url = input_url
    sheet = client.open_by_url(url)
    
    worksheet = sheet.worksheet('Coal_Data')
    coal_df = pd.DataFrame(worksheet.get_all_records())
    coal_df.set_index('Supplier', inplace=True)
    coal_df = coal_df.apply(pd.to_numeric, errors='coerce')

    worksheet = sheet.worksheet('Limit_CFB12')
    limit_cfb12_df = pd.DataFrame(worksheet.get_all_records())
    limit_cfb12_df = limit_cfb12_df.apply(lambda x: x.str.strip()).replace('', np.nan)
    limit_cfb12_df.dropna(inplace=True)

    worksheet = sheet.worksheet('Limit_CFB3')
    limit_cfb3_df = pd.DataFrame(worksheet.get_all_records())
    limit_cfb3_df = limit_cfb3_df.apply(lambda x: x.str.strip()).replace('', np.nan)
    limit_cfb3_df.dropna(inplace=True)

    worksheet = sheet.worksheet('Shipment')
    shipment_df = pd.DataFrame(worksheet.get_all_records())
    shipment_df.set_index('Date', inplace=True)
    shipment_df = shipment_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    shipment_df.dropna(inplace=True)
            
    worksheet = sheet.worksheet('Operation')
    op_df = pd.DataFrame(worksheet.get_all_records())
    op_df.set_index('Date', inplace=True)
    op_df.dropna(inplace=True)

    # _______Function______
    def shift_list(l, idx, shift_idx):
        return l[l.index(idx) + shift_idx]

    def norm(x, min, max):
        return (x-min)/(max-min)

    def sigmoid(x, a, b, c, d):
        return a / (b + math.e**(-c*(x-d)))

    # __Simulation_Config___
    Sim_Week_Duration =  4
    Sim_Week_Inc_Step =  2

    # supplier = coal_df.index.tolist()[:5]
    supplier = coal_df.index.tolist()

    # ______Constants_______
    ratio_step = 10
    n_ratio_step = 100/ratio_step

    Qro12 = 286
    Qro12_norm = norm(Qro12, min=200, max=330)
    Qro3 = 282
    Qro3_norm = norm(Qro3, min=200, max=330)
    T_stack = norm(165, min=120, max=185)
    Excess_O2 = norm(0.025, min=0, max=0.1)
    Pa_ratio = 0.64

    param_list = ['GCV', 'TM', '%H', '%S', '%Ash', '%Fe2O3', '%Na2O']
    param_lookup_list = ['Gross Calorific Value (kcal/kg)', 'Total Moisture (%)', 'Hydrogen', 'Sulphur Content (%)', 'Ash Content (%)', 'Fe2O3', 'Na2O']
    param_min = [17000, 0, 0 , 0, 0]
    param_max = [29000, 36, 10 , 1, 10]

    # ______Dataframes_______
    df = pd.DataFrame(pd.date_range(Start_Date, periods=(Sim_Week_Total * 7) + 1, freq='D'), columns=['Date']).set_index('Date')
    op_df = op_df.loc[[d for d in op_df.index if d in df.index], :]
    shipment_df = shipment_df.loc[[d for d in shipment_df.index if d in df.index], :]

    df.loc[df.index[0]:, [f"OP_CFB{n}" for n in [1, 2, 3]]] = 0
    df.loc[[d for d in op_df.index], [f"OP_CFB{i + 1}" for i in range(3)]] = op_df[[f"CFB{i + 1}" for i in range(3)]].values/24

    df.loc[df.index[0]:, [f"In_{sp}" for sp in supplier]] = 0
    df.loc[[d for d in shipment_df.index], [f"In_{sp}" for sp in supplier]] = shipment_df[[f"{sp}" for sp in supplier]].values

    df[[f"Remain_{sp}" for sp in supplier]] = 0
    df.loc[Start_Date, [f"Remain_{sp}" for sp in supplier]] = coal_df.loc[supplier, 'Coal Remaining (ton)'].tolist()

    # save df_daily for later use
    df_daily = df.copy()
    # print(df_daily)

    # Resample to week
    df = df.reset_index().resample(f"W-{df.index[0].day_name()[:3]}", on='Date').sum()
    # print(df)

    def simulation(df, show_solver_log=False):
        # ___Initialize model___
        m = ConcreteModel()

        # ______Constants_______
        date_rng = df.index.tolist()

        m.zero = Param(default=0)

        # _______Variables______
        m.remain = Var(date_rng, supplier, domain=NonNegativeReals)
        m.remain_notzero = Var(date_rng[1:], supplier, domain=Binary)

        m.cfb12_ratio = Var(date_rng[1:], supplier, domain=NonNegativeReals, bounds=(0, n_ratio_step), initialize=0.1)
        m.cfb3_ratio = Var(date_rng[1:], supplier, domain=NonNegativeReals, bounds=(0, n_ratio_step), initialize=0.1)
        m.cfb12_select = Var(date_rng[1:], supplier, domain=Binary)
        m.cfb3_select = Var(date_rng[1:], supplier, domain=Binary)
        m.cfb12_use = Var(date_rng[1:], supplier, domain=NonNegativeReals)
        m.cfb3_use = Var(date_rng[1:], supplier, domain=NonNegativeReals)

        # ______Equations_______
        m.cons = ConstraintList()

        # total ratio = 100%
        for d in date_rng[1:]:
            m.cons.add(sum(m.cfb12_ratio[d, s] for s in supplier) == n_ratio_step)
            m.cons.add(sum(m.cfb3_ratio[d, s] for s in supplier) == n_ratio_step)

        # if ratio > 0 then select = 1, else if ratio == 0 then select = 0
        for d, s in product(date_rng[1:], supplier):
            m.cons.add(m.cfb12_ratio[d, s] <= n_ratio_step * m.cfb12_select[d, s])
            m.cons.add(m.cfb12_ratio[d, s] >= m.cfb12_select[d, s])
            m.cons.add(m.cfb3_ratio[d, s] <= n_ratio_step * m.cfb3_select[d, s])
            m.cons.add(m.cfb3_ratio[d, s] >= m.cfb3_select[d, s])

        # total number of select not exceed 2
        for d in date_rng[1:]:
            m.cons.add(sum(m.cfb12_select[d, s] for s in supplier) <= 2)
            m.cons.add(sum(m.cfb3_select[d, s] for s in supplier) <= 2)

        # calculate coal mixing parameters
        cfb12_mix = {p: {d: sum(m.cfb12_ratio[d, s] * coal_df.loc[s, param_lookup_list[param_list.index(p)]] for s in supplier) 
                            / sum(m.cfb12_ratio[d, s] for s in supplier) for d in date_rng[1:]} for p in param_list}
        cfb3_mix = {p: {d: sum(m.cfb3_ratio[d, s] * coal_df.loc[s, param_lookup_list[param_list.index(p)]] for s in supplier)
                            / sum(m.cfb3_ratio[d, s] for s in supplier) for d in date_rng[1:]} for p in param_list}
        
        cfb12_mix_norm = {p: {d: norm(cfb12_mix[p][d]*(4.1868 if p == 'GCV' else 1), param_min[param_list.index(p)], param_max[param_list.index(p)]) for d in date_rng[1:]} for p in param_list[:5]}
        cfb3_mix_norm = {p: {d: norm(cfb3_mix[p][d]*(4.1868 if p == 'GCV' else 1), param_min[param_list.index(p)], param_max[param_list.index(p)]) for d in date_rng[1:]} for p in param_list[:5]}
        
        # control mixing parameters in boundary
        for d in date_rng[1:]:
            for p in limit_cfb12_df['Parameter'].tolist():
                x = limit_cfb12_df.loc[limit_cfb12_df['Parameter'] == p]
                m.cons.add((x['Lower Bound'].values[0], cfb12_mix[p][d], x['Upper Bound'].values[0]))
            for p in limit_cfb3_df['Parameter'].tolist():
                x = limit_cfb3_df.loc[limit_cfb3_df['Parameter'] == p]
                m.cons.add((x['Lower Bound'].values[0], cfb3_mix[p][d], x['Upper Bound'].values[0]))
        
        #Correction Tstack when Fe and Na over limit 
        T_Fe12 = {d: (sigmoid(cfb12_mix['%Fe2O3'][d], 5, 1, 40, 11.5) + sigmoid(cfb12_mix['%Fe2O3'][d], 10, 1, 40, 14.5)) for d in date_rng[1:]}
        T_Fe3 = {d: (sigmoid(cfb3_mix['%Fe2O3'][d], 5, 1, 40, 11.5) + sigmoid(cfb3_mix['%Fe2O3'][d], 10, 1, 40, 14.5)) for d in date_rng[1:]}

        T_Na12 = {d: (sigmoid(cfb12_mix['%Na2O'][d], 5, 1, 40, 2.25) + sigmoid(cfb12_mix['%Na2O'][d], 10, 1, 40, 5)) for d in date_rng[1:]}
        T_Na3 = {d: (sigmoid(cfb3_mix['%Na2O'][d], 5, 1, 200, 2.25) + sigmoid(cfb3_mix['%Na2O'][d], 10, 1, 40, 5)) for d in date_rng[1:]}
        
        T12 = {d: norm(165 + T_Fe12[d] + T_Na12[d], min=120, max=185) for d in date_rng[1:]}
        T3 = {d: norm(165 + T_Fe3[d] + T_Na3[d], min=120, max=185) for d in date_rng[1:]}

        # calculate lime flow, boiler eff and coal consumption
        cfb12_lime_flow = {d: 1.098044 - 0.30009*Qro12_norm - 2.205101*cfb12_mix_norm['%S'][d] + 4.485632*(cfb12_mix_norm['%S'][d]**2) for d in date_rng[1:]}
        cfb12_lime_flow_norm = {d: norm(cfb12_lime_flow[d], min=0, max=3.5) for d in date_rng[1:]}
        cfb12_ash_flow = {d: -0.157 + 0.1464*Qro12_norm - 0.1778*cfb12_mix_norm['GCV'][d] + 0.7089*cfb12_lime_flow_norm[d] + 0.423*cfb12_mix_norm['%Ash'][d]
                        - 0.056*Excess_O2 + 0.1962*Pa_ratio for d in date_rng[1:]}
        cfb12_boiler_eff = {d: 98.2894 + 0.2862*Qro12_norm + 5.453*cfb12_mix_norm['GCV'][d] - 4.5807*cfb12_mix_norm['TM'][d]
                            - 17.9721*cfb12_mix_norm['%H'][d] - 0.4126*cfb12_mix_norm['%Ash'][d] - 2.7522*cfb12_lime_flow_norm[d]
                            - 2.9131*T12[d] - 2.6225*Excess_O2 for d in date_rng[1:]}

        cfb3_lime_flow = {d: 1.098044 - 0.30009*Qro3_norm - 2.205101*cfb3_mix_norm['%S'][d] + 4.485632*(cfb3_mix_norm['%S'][d]**2) for d in date_rng[1:]}
        cfb3_lime_flow_norm = {d: norm(cfb3_lime_flow[d], min=0, max=3.5) for d in date_rng[1:]}
        cfb3_ash_flow = {d: -0.157 + 0.1464*Qro3_norm - 0.1778*cfb3_mix_norm['GCV'][d] + 0.7089*cfb3_lime_flow_norm[d] + 0.423*cfb3_mix_norm['%Ash'][d]
                        - 0.056*Excess_O2 + 0.1962*Pa_ratio for d in date_rng[1:]}
        cfb3_boiler_eff = {d: 98.2894 + 0.2862*Qro3_norm + 5.453*cfb3_mix_norm['GCV'][d] - 4.5807*cfb3_mix_norm['TM'][d]
                            - 17.9721*cfb3_mix_norm['%H'][d] - 0.4126*cfb3_mix_norm['%Ash'][d] - 2.7522*cfb3_lime_flow_norm[d]
                            - 2.9131*T3[d] - 2.6225*Excess_O2 for d in date_rng[1:]}

        # calculate coal use
        cfb12_coal_flow = {}
        cfb3_coal_flow = {}
        cfb12_use_sum = {}
        cfb3_use_sum = {}
        for d in date_rng[1:]:
            cfb12_coal_flow[d] = (Qro12*1000) / (cfb12_boiler_eff[d]/100) / (cfb12_mix['GCV'][d]*4.1868)
            cfb3_coal_flow[d] = (Qro3*1000) / (cfb3_boiler_eff[d]/100)  /(cfb3_mix['GCV'][d]*4.1868)

            cfb12_use_sum[d] = cfb12_coal_flow[d] * df.loc[d, ['OP_CFB1', 'OP_CFB2']].sum() * (3600*24) / 1000
            cfb3_use_sum[d] = cfb3_coal_flow[d] * df.loc[d, 'OP_CFB3'] * (3600*24) / 1000

        for d, s in product(date_rng[1:], supplier):
            m.cons.add(m.cfb12_use[d, s] == (m.cfb12_ratio[d, s] / n_ratio_step) * cfb12_use_sum[d])
            m.cons.add(m.cfb3_use[d, s] == (m.cfb3_ratio[d, s] / n_ratio_step) * cfb3_use_sum[d])

        # calculate coal remain
        for d, s in product(date_rng, supplier):
            if d == date_rng[0]:
                m.cons.add(m.remain[d, s] == df.loc[d, f"Remain_{s}"])
            else:
                m.cons.add(m.remain[d, s] == m.remain[shift_list(date_rng, d, -1), s] - m.cfb12_use[d, s] - m.cfb3_use[d, s] + df.loc[d, f"In_{s}"])

        # coal remain >= 0 (exclude incoming shipment)
        for d, s in product(date_rng[1:], supplier):
            m.cons.add(m.remain[shift_list(date_rng, d, -1), s] - m.cfb12_use[d, s] - m.cfb3_use[d, s] >= 0)

        # if remain > 0 then notzero = 1, else if remain == 0 then notzero = 0
        for d, s in product(date_rng[1:], supplier):
            m.cons.add(m.remain[d, s] <= 999999 * m.remain_notzero[d, s])
            m.cons.add(m.remain[d, s] >= m.remain_notzero[d, s])

        # calculate cost
        coal_cost = {d: sum((m.cfb12_use[d, s] + m.cfb3_use[d, s]) * coal_df.loc[s, 'Coal Cost (THB/ton)'] for s in supplier) for d in date_rng[1:]}
        lime_cost = {d: (cfb12_lime_flow[d]*df.loc[d, ['OP_CFB1', 'OP_CFB2']].sum() + cfb3_lime_flow[d]*df.loc[d, 'OP_CFB3']) * (3600*24) / 1000 * lime_price for d in date_rng[1:]}
        flyash_cost = {d: (cfb12_ash_flow[d]*df.loc[d, ['OP_CFB1', 'OP_CFB2']].sum()*0.9 + cfb3_ash_flow[d]*df.loc[d, 'OP_CFB3']*0.9) * (3600*24) / 1000 * flyash_price  for d in date_rng[1:]}
        bottomash_cost = {d: (cfb12_ash_flow[d]*df.loc[d, ['OP_CFB1', 'OP_CFB2']].sum()*0.1 + cfb3_ash_flow[d]*df.loc[d, 'OP_CFB3']*0.1) * (3600*24) / 1000 * bottomash_price  for d in date_rng[1:]}
        fe2o3_cost = {d: cfb12_use_sum[d] * (sigmoid(cfb12_mix['%Fe2O3'][d], 355.34, 1, 40, 14.5)) + 
                    cfb3_use_sum[d] * (sigmoid(cfb3_mix['%Fe2O3'][d], 355.34, 1, 40, 14.5)) for d in date_rng[1:]}
        na2o_cost = {d: cfb12_use_sum[d] * (sigmoid(cfb12_mix['%Na2O'][d], 355.34, 1, 40, 5)) + 
                    cfb3_use_sum[d] * (sigmoid(cfb3_mix['%Na2O'][d], 355.34, 1, 40, 5)) for d in date_rng[1:]}
        remain_penalty = {d: sigmoid(sum(m.remain_notzero[d, s] for s in supplier), 1e8, 1, 100, 3.5) for d in date_rng[1:]}

        # ______Objective_______
        # m.obj = Objective(expr=sum(coal_cost[d] + lime_cost[d] + flyash_cost[d] + bottomash_cost[d] + fe2o3_cost[d] + na2o_cost[d] + remain_penalty[d] for d in date_rng[1:]), sense=minimize)
        m.obj = Objective(expr=sum(lime_cost[d] + flyash_cost[d] + bottomash_cost[d] + fe2o3_cost[d] + na2o_cost[d] + remain_penalty[d] for d in date_rng[1:]), sense=minimize)

        #_____Solve Problem_____
        solver = SolverFactory('bonmin')

        result = solver.solve(m, tee=show_solver_log)

        #_______Results________
        result_df = pd.DataFrame(index=df.index)

        for d in reversed(date_rng):
            if d == date_rng[0]:
                result_df.loc[d, df.columns.tolist()] = df.loc[d, :]

            else:
                result_df.loc[d, 'Cost_Coal'] = coal_cost[d]()
                result_df.loc[d, 'Cost_Lime'] = lime_cost[d]()
                result_df.loc[d, 'Cost_FlyAsh'] = flyash_cost[d]()
                result_df.loc[d, 'Cost_BottomAsh'] = bottomash_cost[d]()
                result_df.loc[d, 'Cost_Fe2O3'] = fe2o3_cost[d]()
                result_df.loc[d, 'Cost_Na2O'] = na2o_cost[d]()
                result_df.loc[d, 'Cost_Total'] = result_df.loc[d, [f"Cost_{x}" for x in ['Coal', 'Lime', 'FlyAsh', 'BottomAsh', 'Fe2O3', 'Na2O']]].sum()
                result_df.loc[d, 'Penalty_Remain'] = remain_penalty[d]()

                result_df.loc[d, [f"Use_{s}" for s in supplier]] = [m.cfb12_use[d, s]() + m.cfb3_use[d, s]() for s in supplier]
                result_df.loc[d, 'Use_Total'] = result_df.loc[d, [f"Use_{s}" for s in supplier]].sum()

                result_df.loc[d, [f"In_{s}" for s in supplier]] = df.loc[d, [f"In_{s}" for s in supplier]]
                result_df.loc[d, 'In_Total'] = result_df.loc[d, [f"In_{s}" for s in supplier]].sum()

                result_df.loc[d, [f"Remain_{s}" for s in supplier]] = [m.remain[d, s]() for s in supplier]
                result_df.loc[d, 'Remain_Total'] = result_df.loc[d, [f"Remain_{s}" for s in supplier]].sum()

                result_df.loc[d, [f"NotZero_{s}" for s in supplier]] = [m.remain_notzero[d, s]() for s in supplier]
                result_df.loc[d, 'NotZero_Total'] = result_df.loc[d, [f"NotZero_{s}" for s in supplier]].sum()

                result_df.loc[d, [f"CFB12_Ratio_{s}" for s in supplier]] = [m.cfb12_ratio[d, s]() for s in supplier]
                result_df.loc[d, 'CFB12_Ratio_Total'] = result_df.loc[d, [f"CFB12_Ratio_{s}" for s in supplier]].sum()
                result_df.loc[d, [f"CFB3_Ratio_{s}" for s in supplier]] = [m.cfb3_ratio[d, s]() for s in supplier]
                result_df.loc[d, 'CFB3_Ratio_Total'] = result_df.loc[d, [f"CFB3_Ratio_{s}" for s in supplier]].sum()

                result_df.loc[d, [f"CFB12_Mixing_{p}" for p in param_list]] = [cfb12_mix[p][d]() for p in param_list]
                result_df.loc[d, [f"CFB3_Mixing_{p}" for p in param_list]] = [cfb3_mix[p][d]() for p in param_list]

                result_df.loc[d, 'CFB12_LimeFlow'] = cfb12_lime_flow[d]()
                result_df.loc[d, 'CFB12_AshFlow'] = cfb12_ash_flow[d]()
                result_df.loc[d, 'CFB12_CoalFlow'] = cfb12_coal_flow[d]()
                result_df.loc[d, 'CFB12_BoilerEff'] = cfb12_boiler_eff[d]()
                result_df.loc[d, 'CFB12_Use'] = cfb12_use_sum[d]()

                result_df.loc[d, 'CFB3_LimeFlow'] = cfb3_lime_flow[d]()
                result_df.loc[d, 'CFB3_AshFlow'] = cfb3_ash_flow[d]()
                result_df.loc[d, 'CFB3_CoalFlow'] = cfb3_coal_flow[d]()
                result_df.loc[d, 'CFB3_BoilerEff'] = cfb3_boiler_eff[d]()
                result_df.loc[d, 'CFB3_Use'] = cfb3_use_sum[d]()

                result_df.loc[d, [f"OP_CFB{n}" for n in [1, 2, 3]]] = df.loc[d, [f"OP_CFB{n}" for n in [1, 2, 3]]]

        return result_df

    def expand_schedule(df, show_solver_log=False):
        df = df.resample('D').bfill()
        
        # ______Constants_______
        date_rng = df.index.tolist()

        ratio_step = 10
        n_ratio_step = 100/ratio_step

        remain = {d: {s: 0.0 for s in supplier} for d in date_rng}
        remain_notzero = {d: {s: 0.0 for s in supplier} for d in date_rng[1:]}

        cfb12_ratio = {d: {s: 0.0 for s in supplier} for d in date_rng[1:]}
        cfb3_ratio = {d: {s: 0.0 for s in supplier} for d in date_rng[1:]}
        cfb12_use = {d: {s: 0.0 for s in supplier} for d in date_rng[1:]}
        cfb3_use = {d: {s: 0.0 for s in supplier} for d in date_rng[1:]}

        # ______Calculation______
        # set remain for day zero to be the same as first day of input df
        for s in supplier:
            remain[date_rng[0]][s] = df.loc[date_rng[0], f"Remain_{s}"]

        # set operation for each day
        for d in date_rng[1:]:
            df.loc[d, ['OP_CFB1', 'OP_CFB2', 'OP_CFB3']] = df_daily.loc[d, ['OP_CFB1', 'OP_CFB2', 'OP_CFB3']]

        # set coal incoming for each day
        for d in date_rng[1:]:
            df.loc[d, [f"In_{s}" for s in supplier]] = df_daily.loc[d, [f"In_{s}" for s in supplier]]

        # calculate integer ratio value for daily operation to achieved approximately weekly blending ratio
        cfb12_ratio_avg = [np.around(df.loc[date_rng[7*(i+1)], [f"CFB12_Ratio_{s}" for s in supplier]].values, 4) for i in range(Sim_Week_Inc_Step)]
        cfb3_ratio_avg = [np.around(df.loc[date_rng[7*(i+1)], [f"CFB3_Ratio_{s}" for s in supplier]].values, 4) for i in range(Sim_Week_Inc_Step)]

        for i, ratio, ratio_avg in zip(range(2), [cfb12_ratio, cfb3_ratio], [cfb12_ratio_avg, cfb3_ratio_avg]):
            # week 1 and week 2 have the same ratio
            if (ratio_avg[0] == ratio_avg[1]).all():
                n_supplier = [(s, ratio_avg[0][supplier.index(s)]) for s in supplier if ratio_avg[0][supplier.index(s)] > 0]

                if len(n_supplier) == 1:
                    for d in date_rng[1:]:
                        ratio[d][n_supplier[0][0]] = n_supplier[0][1]
                else:
                    ratio_set = np.array([x/14 for x in range(14)])
                    q, mod = divmod(n_supplier[0][1], 1)
                    idx = (np.abs(ratio_set - mod)).argmin()

                    for i, d in enumerate(date_rng[1:]):
                        ratio[d][n_supplier[0][0]] = q + (0 if (13 - i) >= idx else 1)
                        ratio[d][n_supplier[1][0]] = n_ratio_step - ratio[d][n_supplier[0][0]]

            # week 1 and week 2 have diffrence ratio
            else:
                for i in range(Sim_Week_Inc_Step):
                    n_supplier = [(s, ratio_avg[i][supplier.index(s)]) for s in supplier if ratio_avg[i][supplier.index(s)] > 0]
                    
                    if len(n_supplier) == 1:
                        for d in date_rng[1+7*i:8+7*(i)]:
                            ratio[d][n_supplier[0][0]] = n_supplier[0][1]
                    else:
                        ratio_set = np.array([x/7 for x in range(7)])
                        q, mod = divmod(n_supplier[0][1], 1)
                        idx = (np.abs(ratio_set - mod)).argmin()

                        for i, d in enumerate(date_rng[1+7*i:8+7*(i)]):
                            ratio[d][n_supplier[0][0]] = q + (0 if (6 - i) >= idx else 1)
                            ratio[d][n_supplier[1][0]] = n_ratio_step - ratio[d][n_supplier[0][0]]
            
        # calculate coal mixing parameters
        cfb12_mix = {p: {d: sum(cfb12_ratio[d][s] * coal_df.loc[s, param_lookup_list[param_list.index(p)]] for s in supplier) 
                            / sum(cfb12_ratio[d][s] for s in supplier) for d in date_rng[1:]} for p in param_list}
        cfb3_mix = {p: {d: sum(cfb3_ratio[d][s] * coal_df.loc[s, param_lookup_list[param_list.index(p)]] for s in supplier)
                            / sum(cfb3_ratio[d][s] for s in supplier) for d in date_rng[1:]} for p in param_list}

        param_min = [17000, 0, 0 , 0, 0]
        param_max = [29000, 36, 10 , 1, 10]
        cfb12_mix_norm = {p: {d: norm(cfb12_mix[p][d]*(4.1868 if p == 'GCV' else 1), param_min[param_list.index(p)], param_max[param_list.index(p)]) for d in date_rng[1:]} for p in param_list[:5]}
        cfb3_mix_norm = {p: {d: norm(cfb3_mix[p][d]*(4.1868 if p == 'GCV' else 1), param_min[param_list.index(p)], param_max[param_list.index(p)]) for d in date_rng[1:]} for p in param_list[:5]}

        #Correction Tstack when Fe and Na over limit
        T_Fe12 = {d: (sigmoid(cfb12_mix['%Fe2O3'][d], 5, 1, 40, 11.5) + sigmoid(cfb12_mix['%Fe2O3'][d], 10, 1, 40, 14.5)) for d in date_rng[1:]}
        T_Fe3 = {d: (sigmoid(cfb3_mix['%Fe2O3'][d], 5, 1, 40, 11.5) + sigmoid(cfb3_mix['%Fe2O3'][d], 10, 1, 40, 14.5)) for d in date_rng[1:]}

        T_Na12 = {d: (sigmoid(cfb12_mix['%Na2O'][d], 5, 1, 40, 2.25) + sigmoid(cfb12_mix['%Na2O'][d], 10, 1, 40, 5)) for d in date_rng[1:]}
        T_Na3 = {d: (sigmoid(cfb3_mix['%Na2O'][d], 5, 1, 200, 2.25) + sigmoid(cfb3_mix['%Na2O'][d], 10, 1, 40, 5)) for d in date_rng[1:]}
        
        T12 = {d: norm(165 + T_Fe12[d] + T_Na12[d], min=120, max=185) for d in date_rng[1:]}
        T3 = {d: norm(165 + T_Fe3[d] + T_Na3[d], min=120, max=185) for d in date_rng[1:]}

        # calculate lime flow, boiler eff and coal consumption
        cfb12_lime_flow = {d: 1.098044 - 0.30009*Qro12_norm - 2.205101*cfb12_mix_norm['%S'][d] + 4.485632*(cfb12_mix_norm['%S'][d]**2) for d in date_rng[1:]}
        cfb12_lime_flow_norm = {d: norm(cfb12_lime_flow[d], min=0, max=3.5) for d in date_rng[1:]}
        cfb12_ash_flow = {d: -0.157 + 0.1464*Qro12_norm - 0.1778*cfb12_mix_norm['GCV'][d] + 0.7089*cfb12_lime_flow_norm[d] + 0.423*cfb12_mix_norm['%Ash'][d]
                        - 0.056*Excess_O2 + 0.1962*Pa_ratio for d in date_rng[1:]}
        cfb12_boiler_eff = {d: 98.2894 + 0.2862*Qro12_norm + 5.453*cfb12_mix_norm['GCV'][d] - 4.5807*cfb12_mix_norm['TM'][d]
                            - 17.9721*cfb12_mix_norm['%H'][d] - 0.4126*cfb12_mix_norm['%Ash'][d] - 2.7522*cfb12_lime_flow_norm[d]
                            - 2.9131*T12[d] - 2.6225*Excess_O2 for d in date_rng[1:]}

        cfb3_lime_flow = {d: 1.098044 - 0.30009*Qro3_norm - 2.205101*cfb3_mix_norm['%S'][d] + 4.485632*(cfb3_mix_norm['%S'][d]**2) for d in date_rng[1:]}
        cfb3_lime_flow_norm = {d: norm(cfb3_lime_flow[d], min=0, max=3.5) for d in date_rng[1:]}
        cfb3_ash_flow = {d: -0.157 + 0.1464*Qro3_norm - 0.1778*cfb3_mix_norm['GCV'][d] + 0.7089*cfb3_lime_flow_norm[d] + 0.423*cfb3_mix_norm['%Ash'][d]
                        - 0.056*Excess_O2 + 0.1962*Pa_ratio for d in date_rng[1:]}
        cfb3_boiler_eff = {d: 98.2894 + 0.2862*Qro3_norm + 5.453*cfb3_mix_norm['GCV'][d] - 4.5807*cfb3_mix_norm['TM'][d]
                            - 17.9721*cfb3_mix_norm['%H'][d] - 0.4126*cfb3_mix_norm['%Ash'][d] - 2.7522*cfb3_lime_flow_norm[d]
                            - 2.9131*T3[d] - 2.6225*Excess_O2 for d in date_rng[1:]}
                            
        # calculate coal use
        cfb12_coal_flow = {}
        cfb3_coal_flow = {}
        cfb12_use_sum = {}
        cfb3_use_sum = {}
        for d in date_rng[1:]:
            cfb12_coal_flow[d] = (Qro12*1000) / (cfb12_boiler_eff[d]/100) / (cfb12_mix['GCV'][d]*4.1868)
            cfb3_coal_flow[d] = (Qro3*1000) / (cfb3_boiler_eff[d]/100)  /(cfb3_mix['GCV'][d]*4.1868)

            cfb12_use_sum[d] = cfb12_coal_flow[d] * df.loc[d, ['OP_CFB1', 'OP_CFB2']].sum() * (3600*24) / 1000
            cfb3_use_sum[d] = cfb3_coal_flow[d] * df.loc[d, 'OP_CFB3'] * (3600*24) / 1000

        for d, s in product(date_rng[1:], supplier):
            cfb12_use[d][s] = (cfb12_ratio[d][s] / n_ratio_step) * cfb12_use_sum[d]
            cfb3_use[d][s] = (cfb3_ratio[d][s] / n_ratio_step) * cfb3_use_sum[d]

        # calculate coal remain
        for d, s in product(date_rng[1:], supplier):
            remain[d][s] = remain[shift_list(date_rng, d, -1)][s] - cfb12_use[d][s] - cfb3_use[d][s] + df.loc[d, f"In_{s}"]

            # force remaining error to 0
            if abs(remain[d][s]) < 1000:
                remain[d][s] = 0

        # if remain > 0 then notzero = 1, else if remain == 0 then notzero = 0
        for d, s in product(date_rng[1:], supplier):
            remain_notzero[d][s] = 1 if remain[d][s] > 0 else 0

        # calculate cost
        coal_cost = {d: sum((cfb12_use[d][s] + cfb3_use[d][s]) * coal_df.loc[s, 'Coal Cost (THB/ton)'] for s in supplier) for d in date_rng[1:]}
        lime_cost = {d: (cfb12_lime_flow[d]*df.loc[d, ['OP_CFB1', 'OP_CFB2']].sum() + cfb3_lime_flow[d]*df.loc[d, 'OP_CFB3']) * (3600*24) / 1000 * lime_price for d in date_rng[1:]}
        flyash_cost = {d: (cfb12_ash_flow[d]*df.loc[d, ['OP_CFB1', 'OP_CFB2']].sum()*0.9 + cfb3_ash_flow[d]*df.loc[d, 'OP_CFB3']*0.9) * (3600*24) / 1000 * flyash_price  for d in date_rng[1:]}
        bottomash_cost = {d: (cfb12_ash_flow[d]*df.loc[d, ['OP_CFB1', 'OP_CFB2']].sum()*0.1 + cfb3_ash_flow[d]*df.loc[d, 'OP_CFB3']*0.1) * (3600*24) / 1000 * bottomash_price  for d in date_rng[1:]}
        fe2o3_cost = {d: cfb12_use_sum[d] * (sigmoid(cfb12_mix['%Fe2O3'][d], 355.34, 1, 40, 14.5)) + 
                    cfb3_use_sum[d] * (sigmoid(cfb3_mix['%Fe2O3'][d], 355.34, 1, 40, 14.5)) for d in date_rng[1:]}
        na2o_cost = {d: cfb12_use_sum[d] * (sigmoid(cfb12_mix['%Na2O'][d], 355.34, 1, 40, 5)) + 
                    cfb3_use_sum[d] * (sigmoid(cfb3_mix['%Na2O'][d], 355.34, 1, 40, 5)) for d in date_rng[1:]}
        remain_penalty = {d: sigmoid(sum(remain_notzero[d][s] for s in supplier), 1e8, 1, 100, 3.5) for d in date_rng[1:]}

        #_______Results________
        for d in date_rng[1:]:
            df.loc[d, 'Cost_Coal'] = coal_cost[d]
            df.loc[d, 'Cost_Lime'] = lime_cost[d]
            df.loc[d, 'Cost_FlyAsh'] = flyash_cost[d]
            df.loc[d, 'Cost_BottomAsh'] = bottomash_cost[d]
            df.loc[d, 'Cost_Fe2O3'] = fe2o3_cost[d]
            df.loc[d, 'Cost_Na2O'] = na2o_cost[d]
            df.loc[d, 'Cost_Total'] = df.loc[d, [f"Cost_{x}" for x in ['Coal', 'Lime', 'FlyAsh', 'BottomAsh', 'Fe2O3', 'Na2O']]].sum()
            df.loc[d, 'Penalty_Remain'] = remain_penalty[d]

            df.loc[d, [f"Use_{s}" for s in supplier]] = [cfb12_use[d][s] + cfb3_use[d][s] for s in supplier]
            df.loc[d, 'Use_Total'] = df.loc[d, [f"Use_{s}" for s in supplier]].sum()

            df.loc[d, 'In_Total'] = df.loc[d, [f"In_{s}" for s in supplier]].sum()

            df.loc[d, [f"Remain_{s}" for s in supplier]] = [remain[d][s] for s in supplier]
            df.loc[d, 'Remain_Total'] = df.loc[d, [f"Remain_{s}" for s in supplier]].sum()

            df.loc[d, [f"NotZero_{s}" for s in supplier]] = [remain_notzero[d][s] for s in supplier]
            df.loc[d, 'NotZero_Total'] = df.loc[d, [f"NotZero_{s}" for s in supplier]].sum()

            df.loc[d, [f"CFB12_Ratio_{s}" for s in supplier]] = [cfb12_ratio[d][s] for s in supplier]
            df.loc[d, 'CFB12_Ratio_Total'] = df.loc[d, [f"CFB12_Ratio_{s}" for s in supplier]].sum()
            df.loc[d, [f"CFB3_Ratio_{s}" for s in supplier]] = [cfb3_ratio[d][s] for s in supplier]
            df.loc[d, 'CFB3_Ratio_Total'] = df.loc[d, [f"CFB3_Ratio_{s}" for s in supplier]].sum()

            df.loc[d, [f"CFB12_Mixing_{p}" for p in param_list]] = [cfb12_mix[p][d] for p in param_list]
            df.loc[d, [f"CFB3_Mixing_{p}" for p in param_list]] = [cfb3_mix[p][d] for p in param_list]

            df.loc[d, 'CFB12_LimeFlow'] = cfb12_lime_flow[d]
            df.loc[d, 'CFB12_AshFlow'] = cfb12_ash_flow[d]
            df.loc[d, 'CFB12_CoalFlow'] = cfb12_coal_flow[d]
            df.loc[d, 'CFB12_BoilerEff'] = cfb12_boiler_eff[d]
            df.loc[d, 'CFB12_Use'] = cfb12_use_sum[d]

            df.loc[d, 'CFB3_LimeFlow'] = cfb3_lime_flow[d]
            df.loc[d, 'CFB3_AshFlow'] = cfb3_ash_flow[d]
            df.loc[d, 'CFB3_CoalFlow'] = cfb3_coal_flow[d]
            df.loc[d, 'CFB3_BoilerEff'] = cfb3_boiler_eff[d]
            df.loc[d, 'CFB3_Use'] = cfb3_use_sum[d]

        return df

    # Simulation
    n_sim = math.ceil((float(Sim_Week_Total) - float(Sim_Week_Duration)) / float(Sim_Week_Inc_Step) + 1)
    for i in tqdm(range(n_sim + 1)):
        url = result_url

        # Find a workbook by url
        sheet = client.open_by_url(url)
        worksheet = sheet.worksheet('Status')
        worksheet.update(f"A{i + 1}", f"Iter {i + 1}/{n_sim + 1}")
        worksheet.update(f"B{i + 1}", datetime.now().strftime(f"%Y-%m-%d"))
        worksheet.update(f"C{i + 1}", datetime.now().strftime(f"%H:%M:%S"))

        if i <= n_sim:
            # Extract part of df for only simalation duration and simulate
            df_week = df.iloc[Sim_Week_Inc_Step * i:Sim_Week_Inc_Step * i + Sim_Week_Duration + 1, :]
            # print(df_week)

            result_week = simulation(df_week, show_solver_log=False)
            # print(result_week)

            result_daily = expand_schedule(result_week.iloc[0:Sim_Week_Inc_Step + 1], True)
            # print(result_daily)

            # Update remaining coal to main df
            df.loc[result_daily.index.tolist()[-1], [f"Remain_{s}" for s in supplier]] = result_daily.iloc[-1][[f"Remain_{s}" for s in supplier]].values

            # Save result to result_df
            if i == 0:
                result_df_daily = result_daily
                result_df_week = result_week.iloc[0:Sim_Week_Inc_Step + 1]
            else:
                result_df_daily = result_df_daily.append(result_daily.tail(Sim_Week_Inc_Step * 7))
                result_df_week = result_df_week.append(result_week.iloc[1:Sim_Week_Inc_Step + 1])

        else:
            result_daily = expand_schedule(result_week.iloc[-(Sim_Week_Inc_Step + 1):], True)
            result_df_daily = result_df_daily.append(result_daily.tail(Sim_Week_Inc_Step * 7))
            result_df_week = result_df_week.append(result_week.tail(Sim_Week_Inc_Step))
            
        sheet = client.open_by_url(url)
        worksheet = sheet.worksheet('Result')
        worksheet.clear()
        df_export = result_df_daily.copy()
        df_export = df_export.reset_index().fillna(0)
        df_export['Date'] = df_export['Date'].map(lambda dt: dt.strftime(f"%Y-%m-%d"))
        worksheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
            
    # Export result data
    # result_df_daily.to_excel('result.xlsx', sheet_name='Result')
    # sheet = client.open_by_url(url)
    # worksheet = sheet.worksheet('Result')
    # worksheet.clear()
    # df_export = result_df_daily.copy()
    # df_export = df_export.reset_index().fillna(0)
    # df_export['Date'] = df_export['Date'].map(lambda dt: dt.strftime(f"%Y-%m-%d"))
    # worksheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
