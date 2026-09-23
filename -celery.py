# -*- coding: utf-8 -*-
"""
Description:  celery配置
"""
import os
import django
from celery import Celery
from django.conf import settings

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

celery_app = Celery('backend')
# celery_app = Celery('TemplateOCR')
celery_app.config_from_object('django.conf:settings')
celery_app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)

