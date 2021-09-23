from django.conf import settings
from django.shortcuts import redirect, render
from django.views import generic

from .models import *


def redirect_home(request):
    return redirect('coal_blending:data_input')


class DataInputView(generic.TemplateView):
    template_name = 'coal_blending/data_input.html'
    
    def get_context_data(self, **kwargs):
        context = super(DataInputView, self).get_context_data(**kwargs)
        
        url = 'https://docs.google.com/spreadsheets/d/1CfxOcwVLGeTAQLGysozopqI2oqZDC4MkrYyumZnIKHw/edit#gid=486638920'
        context['link'] = url

        # use creds to create a google client to interact with the Google Drive API
        json_creds = config('GOOGLE_APPLICATION_CREDENTIALS')
        creds_dict = json.loads(json_creds)
        creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
        scopes = ['https://spreadsheets.google.com/feeds'] 
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scopes)
        client = gspread.authorize(creds)

        # Find a workbook by url
        sheet = client.open_by_url(url)
        worksheet = sheet.worksheet('Coal_Data')
        coal_df = pd.DataFrame(worksheet.get_all_records())
        coal_df.set_index('Supplier', inplace=True)
        coal_df = coal_df.astype('float')
        context['df1'] = coal_df.to_html(table_id='Table1', classes='table table-bordered table-hover table-responsive')
        
        worksheet = sheet.worksheet('Limit_CFB12')
        limit_cfb12_df = pd.DataFrame(worksheet.get_all_records())
        limit_cfb12_df.dropna(inplace=True)
        context['df2'] = limit_cfb12_df.to_html(table_id='Table2', classes='table table-bordered table-hover table-responsive')
        
        worksheet = sheet.worksheet('Limit_CFB3')
        limit_cfb3_df = pd.DataFrame(worksheet.get_all_records())
        limit_cfb3_df.dropna(inplace=True)
        context['df3'] = limit_cfb3_df.to_html(table_id='Table3', classes='table table-bordered table-hover table-responsive')
        
        worksheet = sheet.worksheet('Shipment')
        shipment_df = pd.DataFrame(worksheet.get_all_records())
        shipment_df.set_index('Date', inplace=True)
        shipment_df.dropna(inplace=True)
        context['df4'] = shipment_df.to_html(table_id='Table4', classes='table table-bordered table-hover table-responsive')
                
        worksheet = sheet.worksheet('Operation')
        op_df = pd.DataFrame(worksheet.get_all_records())
        op_df.set_index('Date', inplace=True)
        op_df.dropna(inplace=True)
        context['df5'] = op_df.to_html(table_id='Table5', classes='table table-bordered table-hover table-responsive')
        
        https://raw.githubusercontent.com/paroonk/heroku-coal-blending/master/bonmin
        
        from sys import executable
        import numpy as np
        import pandas as pd
        from itertools import product
        from pyomo.environ import *

        pd.options.display.float_format = "{:,.2f}".format
        pd.options.display.max_columns = None
        pd.options.display.expand_frame_repr = False

        items = {
            'hammer':       {'weight': 5, 'benefit': 8},
            'wrench':       {'weight': 7, 'benefit': 3},
            'screwdriver':  {'weight': 4, 'benefit': 6},
            'towel':        {'weight': 3, 'benefit': 11},
        }
        weight_max = 14

        # ___Initialize model___
        m = ConcreteModel()

        # ______Constants_______


        # ______Variables_______
        m.x = Var(items, domain=Binary)

        # ______Equations_______
        m.cons = ConstraintList()
        m.cons.add(sum(items[i]['weight'] * m.x[i] for i in items) <= weight_max)

        # ______Objective_______
        m.obj = Objective(expr=sum(items[i]['benefit'] * m.x[i] for i in items), sense=maximize)

        #_____Solve Problem_____
        solver = SolverFactory(executable='apopt.py')
        results = solver.solve(m, tee=False)
        
        context['test'] = m.obj()
                
        return context
