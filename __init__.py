from __future__ import absolute_import, unicode_literals
# from backend.backend.celery import celery_app
from backend.celery import celery_app

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

__all__=['celery_app']