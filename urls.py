# -*- coding: utf-8 -*-
"""
----------------------------------------------------------------------
    File Name:  urls
    Description:    
    date:  2022/5/30 0030 16:29
----------------------------------------------------------------------
"""

from django.contrib.staticfiles.urls import staticfiles_urlpatterns,static

from django.urls import path, re_path, include

from . import views
from backend import settings

urlpatterns = [
    path(r'template/create', views.create_project_view),
    path(r'template/create_model_type', views.create_model_type),
    path(r'template/update1', views.project_update),
    path(r'template/update_moudel', views.project_update_moudel),
    path('template/delete', views.project_delete),
    path('template/model_delete', views.model_delete),
    path('template/file_upload', views.file_upload),
    path('template/create_mark', views.create_mark),
    path('template/project_list', views.get_project_list),
    path('template/mark_list', views.get_mark_list),
    path('template/mark_list1', views.get_mark_list1),
    path('template/update_mark', views.mark_update),
    path('template/delete_by_mark_id', views.del_point_by_mark_id),
    path('template/delete_mark', views.mark_delete),
    path('template/delete_markresults', views.markresults_delete),
    path('template/update_markresults', views.markresults_update),
    path('template/get_img_by_project_id', views.get_img_by_project_id),
    # path('template/mark_points', views.get_mark_by_project_id),
    path('template/get_mark_by_project_id', views.get_mark_by_project_id),
    path('template/get_mark_by_Moudeltype_id', views.get_mark_by_Moudeltype_id),
    path('template/get_mark_points', views.get_mark_points),
    path('template/add_points', views.add_points),
    path('template/projects', views.get_project),
    path('template/model_predict',views.get_model_predict),
    path('template/model_uploaddata2db',views.uploadData2DB),
    # path('template/model_uploaddata2db',views.uploadData2DB),
    path('template/model_downloaddata',views.model_downloaddata),
    path('template/project_list_modeltype',views.get_project_list_modeltype),
    path('template/model_list',views.get_model_list),
    path('template/model_list1',views.get_model_list1),
    path('template/mark_list_train',views.get_mark_list_train),
    path('template/data_to_train',views.data_to_train),
    # path('template/celery_index', views.celery_index),
    # path('template/celery_index2', views.celery_index2),
    path(r'template/training_new_model',views.training_new_model ),
    path(r'template/get_file_update_time',views.get_file_update_time ),
    path(r'template/text_recognize',views.text_recognize ),
    path(r'template/upload_img',views.upload_img ),
    path(r'template/bulk_predictor',views.bulk_predictor ),
    path(r'template/get_bulk_res',views.get_bulk_res),
    path(r'template/delete_all_bulk',views.delete_all_bulk ),
    path(r'template/export_bulk',views.export_bulk ),
    path(r'template/update_bulk',views.update_bulk ),
    path(r'template/bulk_delete',views.bulk_delete ),
    path(r'template/delete_all_trainset',views.delete_all_trainset ),
    path(r'template/contractdata',views.contractData ),
    path(r'template/getprogress',views.get_training_status ),
    path(r'template/resetmodel',views.resetmodel ),
    path(r'template/refresh_model_status', views.refresh_model_status),
    path(r'template/bulk_predictor_http', views.bulk_predictor_http),
    # path('',include('celerybar.urls')),
    # path('celery-progress/', include('celery_progress.urls')),

    re_path(r'template/build_model_url/(\d+)/$', views.build_model_url),
    re_path(r'template/api/(\d+)$', views.api),
    re_path(r'template/show_data', views.show_data),

    # path('test', views.test)
]
urlpatterns += staticfiles_urlpatterns()
urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)

print('【check 1 st】',settings.MEDIA_URL,settings.MEDIA_ROOT)

print('【check urlpatterns】',urlpatterns)