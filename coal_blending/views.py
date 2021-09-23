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

        # use creds to create a google client to interact with the Google Drive API
        json_creds = config('GOOGLE_APPLICATION_CREDENTIALS')
        creds_dict = json.loads(json_creds)
        creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
        scopes = ['https://spreadsheets.google.com/feeds'] 
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scopes)
        client = gspread.authorize(creds)

        # Find a workbook by url
        url = 'https://docs.google.com/spreadsheets/d/1CfxOcwVLGeTAQLGysozopqI2oqZDC4MkrYyumZnIKHw/edit#gid=486638920'
        sheet = client.open_by_url(url)
        worksheet = sheet.worksheet('Coal_Data')
        df = pd.DataFrame(worksheet.get_all_records())
        
        context['df'] = df.to_html(table_id='dbTable', classes='table table-bordered table-hover table-responsive')
        
        context['link'] = url
        
        return context
