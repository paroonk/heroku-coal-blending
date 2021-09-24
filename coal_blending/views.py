
from django.conf import settings
from django.shortcuts import redirect, render
from django.views import generic


from .models import *
from .optimize import *


def redirect_home(request):
    return redirect('coal_blending:data_input')


class DataInputView(generic.TemplateView):
    template_name = 'coal_blending/data_input.html'
    
    def get_context_data(self, **kwargs):
        context = super(DataInputView, self).get_context_data(**kwargs)
        
        url = 'https://docs.google.com/spreadsheets/d/1rflUWAGZd0vlKGxEGR_kn7LuVmegOQiL8c35Nzh6lmM/edit#gid=486638920'
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
                
        return context


class ResultView(generic.TemplateView):
    template_name = 'coal_blending/result.html'
    
    def get_context_data(self, **kwargs):
        context = super(ResultView, self).get_context_data(**kwargs)
        
        url = 'https://docs.google.com/spreadsheets/d/1N2EiCGQMSnxOmzuFyE6cnwrLoBTLdrjF-h7l2cUpAo0/edit#gid=979337795'
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
        worksheet = sheet.worksheet('Updated')
        worksheet.update('A1', str(datetime.now()))
                
        return context