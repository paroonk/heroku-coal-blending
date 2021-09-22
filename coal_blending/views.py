from django.conf import settings
from django.shortcuts import redirect, render
from django.views import generic

from .models import *


def redirect_home(request):
    return redirect('coal_blending:dashboard')


class DashboardView(generic.TemplateView):
    template_name = 'coal_blending/dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super(DashboardView, self).get_context_data(**kwargs)
        context['test'] = 'Test123'
        return context