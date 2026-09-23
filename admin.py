from django.contrib import admin

# Register your models here.
from . import models

admin.site.register(models.Project)
admin.site.register(models.Picture)
admin.site.register(models.Mark)
admin.site.register(models.Moudeltype)
admin.site.register(models.Markresults)

