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
        
        url = input_url
        context['link'] = url

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


def trigger_optimizer(request):
    coal_optimize.delay()
    
    return redirect(request.META.get('HTTP_REFERER'))

class ResultView(generic.TemplateView):
    template_name = 'coal_blending/result.html'
    
    def get_context_data(self, **kwargs):
        context = super(ResultView, self).get_context_data(**kwargs)
        
        url = 'https://docs.google.com/spreadsheets/d/1N2EiCGQMSnxOmzuFyE6cnwrLoBTLdrjF-h7l2cUpAo0/edit#gid=979337795'
        context['link'] = url
                
        return context
