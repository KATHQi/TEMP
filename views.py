# -*- coding: utf-8 -*-
import json
from io import BytesIO
import base64
import urllib.request
import ssl
from rest_framework import status
from django.db.models import Max
# from django.db import transaction
import copy
import re
import cv2
import fitz
from django.utils import timezone
from django.core.paginator import Paginator
from django.core.files.uploadhandler import InMemoryUploadedFile, TemporaryUploadedFile
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Mark, Project, Picture, Moudeltype,Markresults,Markresults_bulk,Bulk_picture,Exportresult_bulk,Bulk_pdf, TrainTask
from .forms import ProjectForm, MarkForm, MarkresultsForm, MoudeltypeForm,Markresults_bulkForm,Bulk_pictureForm,Exportresult_bulkFrom
# from .onnx_ocr.onnx_ocr import recognize
from .serializers import ProjectSerializer, MarkSerializer, ProjectWithImage, PictureSerializer, ProjectWithMark,MoudeltypeSerializer,ModelWithImage,MarkresultsSerializer,Exportresult_bulkSerializer
from PIL import Image, ImageDraw, ExifTags
import numpy as np
import shutil
import matplotlib.pyplot as plt
from time import sleep
from time import localtime,strftime
import datetime

import sys,os
import paddle
from Funcs.text_det_ser import text_pred_res_det,text_pred_res_bulk2
from django.shortcuts import render
from django.shortcuts import HttpResponse
from task import async_task, REC_Model,DET_Model, recognizer,generat_class_file#,DET_retrainer,REC_MODEL_DIR,CLS_MODEL_DIR #,SerPredictor_Model,TextSystem_Model
from Funcs.img_preprocessing import redink_remover,deskew,rotateImage
from Funcs.calculators import char_overlap_rate

import importlib
from Funcs.bulkrecfuncs.modelfunc import SER_generator,kie_predictor ,text_sys ,convert_pdf_with_adaptive_size, clear_directory,reload_model
from Funcs.bulkrecfuncs.filefuncs import clear_directory
from urllib.parse import unquote
import sqlite3
import yaml

from rest_framework.decorators import api_view
from django.views.decorators.csrf import csrf_exempt
from rest_framework.response import Response

# import threading
import subprocess
import atexit
import signal
import pandas as pd





text_recognizer=REC_Model()
text_detector=DET_Model()
# SerPredictor=kie_predictor #SerPredictor_Model()
# text_sys=text_sys  #TextSystem_Model()
child_proc=None


train_status={'state':'pending','progress':0, 'error':''}

# def run_training_pipeline(pretrained_path,data_path):
#     global train_status
#     try:
#         train_status['state']='running'
#         train_status['progress']=1
#         generat_class_file()
#         train_status['progress']=3
#         DET_retrainer(pretrained_path,data_path)
#         train_status['state']='finished'
#         train_status['progress']=100
#     except Exception as e:
#         train_status['state']='error'
#         train_status['error']=str(e)

def cleanup():
    global proc
    print("主程序退出，杀掉子进程组...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        update_status(0,status='pending',progress=0)
    except Exception as e:
        print("无法杀掉子进程组:", e)

def update_status(task_id, status=None, progress=None, error=None):
    task = TrainTask.objects.get(id=task_id)
    if status: task.status = status
    if progress is not None: task.progress = progress
    if error: task.error_msg = error
    task.last_update = timezone.now()
    task.save()
update_status(0,status='pending',progress=0)

def point_trans(data):
    x = data[0][0]
    y = data[0][1]
    # x=(data[0][0]+data[1][0]+data[2][0]+data[3][0])/4
    # y=(data[0][1]+data[1][1]+data[2][1]+data[3][1])/4
    width=abs(data[0][0]-data[1][0])
    height=abs(data[1][1]-data[2][1])

    return {'x':x,'y':y,'width':width,'height':height}


def transPoint(points):

    x1 = int(points['x'])
    y1 = int(points['y'])
    x2 = int(points['x'] + (points['width']))
    y2 = y1
    x3 = x2
    y3 = int(points['y'] + (points['height']))
    x4 = x1
    y4 = y3

    return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

def draw_boxes(image, boxes):
    # if not image:
    #     return image
    # print('[imageshape]',image.shape)
    for box in boxes:
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    return image

def getMarkid(pred,points,create_time,last_mod_time,project_id,moudeltype_id):
    # 优先在相同 moudeltype 下查找该 mark_name，避免不同模型类型的同名标签冲突
    res = Mark.objects.filter(mark_name=pred, moudeltype_id=moudeltype_id).values()
    # maxid=Mark.objects.all().aggregate(Max('id'))['id__max']
    # xx=Mark.objects.filter(mark_name='others').values()#[0]['id']
    # temp = {'id': maxid + 1, 'mark_name': pred, 'points': {}, 'create_time': create_time,
    #         'last_mod_time': last_mod_time, 'flag': 1, 'mark_type': 'TX', 'project_id': project_id,'moudeltype_id':moudeltype_id}
    # form_ = MarkForm(temp)
    # marks = form_.save(False)
    # marks.project_id=project_id
    # marks.id=maxid + 1
    # marks.moudeltype_id=moudeltype_id
    # marks.save(True)
    # return maxid+1
    print('res',res)
    print('project_id',project_id)
    print('moudeltype_id',moudeltype_id)
    if len(res)>=1:
        return res[0]['id']
    else:
        maxid=Mark.objects.all().aggregate(Max('id'))['id__max']
        xx=Mark.objects.filter(mark_name='others').values()#[0]['id']
        temp = {'id': maxid + 1, 'mark_name': pred, 'points': {}, 'create_time': create_time,
                'last_mod_time': last_mod_time, 'flag': 1, 'mark_type': 'TX', 'project_id': project_id,'moudeltype_id':moudeltype_id}
        form_ = MarkForm(temp)
        marks = form_.save(False)
        marks.project_id=project_id
        marks.id=maxid + 1
        marks.moudeltype_id=moudeltype_id
        marks.save(True)
        return maxid+1
def resize_if_needed(img, max_edge=1000):
    """
    如果图像的最长边 > max_edge，则按等比缩小图像
    :param img: OpenCV 图像 (numpy array)
    :param max_edge: 最长边的最大值，默认 1000
    :return: 缩放后的图像（或原图）
    """
    h, w = img.shape[:2]
    max_dim = max(h, w)
 
    if max_dim <= max_edge:
        return img  # 无需缩放

    # 计算缩放比例，并应用
    scale = max_edge / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_img

@api_view(['POST'])
def create_model_type(request):
    

    if request.method != "POST":
        form = ProjectForm()
    else:

        maxid_ = Moudeltype.objects.all().aggregate(Max('model_type'))['model_type__max']
        if maxid_==None:
            maxid_=0
        temp = {'model_type': maxid_+1, 'model_type_name': request.query_params['model_type_name'],
                'model_example_pic': request.data['picture']['picture'][7:]}

        form = MoudeltypeForm(temp)
        try:
            origin_img = request.data['picture']['picture'][7:]#request.data['model_example_pic'][7:]#
            print('\norigin_img',origin_img)
            picture = Picture.objects.filter(picture=origin_img).first()
        except Exception:
            picture = None
        if form.is_valid():
            moudeltype = form.save(False)
            moudeltype.model_type=maxid_+1
            moudeltype.save(True)

            if picture:
                picture.picture_name = picture.picture.name
                picture.moudeltype = moudeltype
                picture.save()
            try:
                print("【moudeltype.model_type_name】", moudeltype.model_type_name)
                # project = Project.objects.get(project_name=f"{moudeltype.model_type_name}")
                
                Mark.objects.create(
                        mark_name="others",
                        
                        mark_type="TX",
                        moudeltype=moudeltype
                    )
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                print("【1Mark 创建失败】", traceback.format_exc())
    # mark_form = MarkForm(data)
    # if mark_form.is_valid():
    #     try:
    #         mark = mark_form.save(False)
    #         mark.moudeltype_id = data['model_type_name']
    #         mark.save(True)
    #         return Response({'code': 200, 'message': '添加mark成功！'})
    #     except Exception as e:
    #         print(e)
    #         return Response({'code': 500, 'message': '插入值失败！'})



            return Response({'code': 200, 'message': '模板创建成功！'})
        else:
            print('创建失败')
            model_type_name = request.query_params.get('model_type_name', '')
            
            # 检查模型名称长度
            if len(model_type_name) > 200:
                return Response({'code': 400, 'form': {}, 'message': '模型名称长度不能超过200个字符！'}, status=400)
            
            # 检查模型名称是否已存在
            if model_type_name and Moudeltype.objects.filter(model_type_name=model_type_name).exists():
                return Response({'code': 400, 'form': {}, 'message': '模板名称已存在！'}, status=400)
            
            # 检查图片
            if picture==None:
                return Response({'code': 400, 'form': form, 'message': '模板创建失败！'}, status=400)
            
            # 其他表单验证错误
            form_errors = form.errors.as_json() if hasattr(form, 'errors') else {}
            return Response({'code': 400, 'form': form_errors, 'message': '表单验证失败，请检查输入数据！'}, status=400)

@api_view(["delete"])
def model_delete(request):
    id = request.query_params['id']
    project = Moudeltype.objects.get(model_type=id)
    try:
        project.picture.picture.delete()
    except Exception as e:
        print("删除图片失败", e)
    project.delete()
    return Response({'code': 200, 'data': 'success'})


# @api_view(['POST'])
# def create_project_view(request):
#     print(111)
#     if request.method != "POST":
#         form = ProjectForm()
#     else:
#         form = ProjectForm(request.query_params)
#         try:
#             origin_img = request.data['picture']['picture'][7:]
#             picture = Picture.objects.filter(picture=origin_img).first()
#         except Exception:
#             picture = None
#             import traceback
#             print(traceback.format_exc())
#             return Response({'code':500, 'form':traceback.format_exc(), 'message': '模板创建失败！'})
#         if form.is_valid():
#             # project = form.save(False)
#             test = Moudeltype.objects.get(model_type=request.query_params['model_type_name'])
#             project = form.save(False)
#             project.model_type = test
#             project.model_type_id = request.query_params['model_type_name']
#             project.model_type_name = test.model_type_name
#             project.save(True)

#             if picture:
#                 picture.picture_name = picture.picture.name
#                 picture.project = project
#                 picture.save()

#             return Response({'code': 200, 'message': '模板创建成功！'})
#         else:
#             print('创建失败')
            
#             return Response({'code':500, 'form':form, 'message': '模板创建失败！'})




@api_view(['POST'])
def create_project_view(request):
    try:
        print('开始创建图像库数据')
        
        # 检查必要的参数
        model_type_name = request.query_params.get('model_type_name')
        if not model_type_name:
            return Response({'code': 400, 'message': '缺少模型类型参数！'}, status=400)
        
        # 验证模型类型是否存在
        try:
            moudeltype = Moudeltype.objects.get(model_type=model_type_name)
        except Moudeltype.DoesNotExist:
            return Response({'code': 404, 'message': '指定的模型类型不存在！'}, status=404)
        
        # 正确获取查询参数，避免列表格式问题
        form_data = {
            'project_name': request.query_params.get('project_name', ''),
            'project_desc': request.query_params.get('project_desc', ''),
            'model_type_name': request.query_params.get('model_type_name', ''),
        }
        
        print(f'Debug form_data: {form_data}')  # 调试信息
        
        form = ProjectForm(form_data)
        
        # 取 picture
        picture = None
        try:
            origin_img = None
            if 'picture' in request.data:
                if request.data['picture'] and 'picture' in request.data['picture']:
                    origin_img = request.data['picture']['picture']
                    if origin_img and origin_img.startswith('/media/'):
                        origin_img = origin_img[7:]
            elif 'picture' in request.query_params:
                pic_raw = request.query_params['picture']
                if pic_raw:
                    try:
                        pic_json = json.loads(pic_raw)
                        pic_path = pic_json.get('picture', '')
                        if pic_path and pic_path.startswith('/media/'):
                            origin_img = pic_path[7:]
                    except (json.JSONDecodeError, ValueError):
                        if pic_raw.startswith('/media/'):
                            origin_img = pic_raw[7:]
                        else:
                            origin_img = pic_raw
            
            if origin_img:
                picture = Picture.objects.filter(picture=origin_img).first()
                
        except Exception as e:
            print(f'图片处理异常: {str(e)}')
            # 图片处理出错不应该阻止项目创建，继续执行
        
        if form.is_valid():
            project = form.save(False)
            project.model_type = moudeltype
            project.model_type_id = model_type_name
            project.model_type_name = moudeltype.model_type_name
            project.save(True)
            
            # 关联图片
            if picture:
                try:
                    picture.picture_name = picture.picture.name
                    picture.project = project
                    picture.save()
                    print(f'成功关联图片: {picture.picture_name}')
                except Exception as e:
                    print(f'图片关联失败: {str(e)}')
                    # 图片关联失败不影响项目创建
            
            return Response({'code': 200, 'message': '图像库数据创建成功！'})
        else:
            print('表单验证失败:', form.errors)
            # 处理常见的验证错误
            error_messages = []
            for field, errors in form.errors.items():
                for error in errors:
                    if 'already exists' in str(error) or '已存在' in str(error):
                        error_messages.append('图像名称已存在，请使用其他名称')
                    else:
                        error_messages.append(f'{field}: {error}')
            
            message = '; '.join(error_messages) if error_messages else '表单验证失败！'
            return Response({'code': 400, 'form': form.errors, 'message': message}, status=400)
            
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print('创建图像库数据时发生错误:', error_msg)
        return Response({'code': 500, 'message': f'服务器错误: {str(e)}'}, status=500)



@csrf_exempt
@api_view(['POST'])
# def file_upload(request): 
#     print('img uploading')
    
#     return Response({'code': 200, 'message': '图片test'})

def file_upload(request):
    
    
    try:
        pic_file = request.FILES.get('file')

        picture = Picture()

        pic_io = BytesIO()
        img = Image.open(pic_file)
        # img_rotate = img
        if img.height>2500 or img.width>2500 :
            img.thumbnail((int(img.width // 1.5), int(img.height // 1.5)))
        img_rotate = rotate(img)


        img_rotate.save(pic_io, img.format)

        pic = InMemoryUploadedFile(
            file=pic_io,
            field_name=None,
            name=pic_file.name,
            content_type=pic_file.content_type,
            size=img_rotate.size,
            charset=None
        )

        picture.picture = pic
        picture.picture_name = picture.picture.name
        picture.save()
        res = PictureSerializer(picture)
        data = {'id': int(res.data['id']), 'url': str(res.data['picture'])}
        print('【check img save path】',data)
        return Response({'code': 200, 'data': data})
    except Exception as e:
        # print('aaaaaaaaaaa ',e)
        import traceback
        print(traceback.format_exc())
        return Response({'code': 403, 'message': '图片上传失败'+str(e)}, status=403)



@api_view(["GET"])
def get_model_list(request):
    page = request.GET.get('pageNum')
    pageSize = request.GET.get('pageSize')
    if page==None:
        page=1
    if pageSize==None:
        pageSize=100
    modellist=Moudeltype.objects.all()
    # 创建分页对象
    ptr = Paginator(modellist, pageSize)
    # 分页
    masters = ptr.page(page)
    # 序列化
    project_serialize = ModelWithImage(masters, many=True)
    # project_serialize=MoudeltypeSerializer(masters, many=True)
    res = {}
    res['code'] = 20000
    res['msg'] = ''
    res['data'] = {'pageNum': int(page), 'pageSize': int(pageSize), 'total': modellist.count(),
                   'records': project_serialize.data}
    return Response(res)


# @api_view(["GET"])
# def get_model_list1(request):

#     project_list = Moudeltype.objects.exclude(model_type='-1')  # .all()#.filter(id!=-1)
#     if request.GET.get('model_type_name'):
#         project_list = project_list.filter(model_type_name=request.GET.get('model_type_name'))  # .all()#.filter(id!=-1)

#     projects =MoudeltypeSerializer(project_list, many=True)
#     data = []
#     temp = []
#     for project in projects.data:
#         res=project['model_type_name']
#         if res in temp:
#             continue
#         temp.append(res)
#         data.append({'value': project['model_type'], 'label': res})

#     return Response({'code': 200, 'data': data})
#     return Response(res)

@api_view(["GET"]) 
def get_model_list1(request):
    project_list = Moudeltype.objects.exclude(model_type='-1')

    if request.GET.get('model_type_name'):
        project_list = project_list.filter(model_type_name=request.GET.get('model_type_name'))

    projects = MoudeltypeSerializer(project_list, many=True)

    data = []
    temp = set()
    for project in projects.data:
        res = project['model_type_name']
        if res in temp:
            continue
        temp.add(res)
        data.append({'value': project['model_type'], 'label': res})

    return  Response({'code': 200, 'data': data})  # ⚠️ 直接返回数组



@api_view(["GET"])
def get_project_list(request):
    page = request.GET.get('pageNum')
    pageSize = request.GET.get('pageSize')
    if not page:
        page=1
    if not pageSize:
        pageSize=1000
    modeltype_filter=request.GET.get('model_type_name')
    project_list = Project.objects.exclude(id='-1')#.all()#.filter(id!=-1)
    if modeltype_filter:
        project_list = project_list.filter(model_type=modeltype_filter)
    # 创建分页对象
    ptr = Paginator(project_list, pageSize)
    # 分页
    masters = ptr.page(page)
    # 序列化
    project_serialize = ProjectWithImage(masters, many=True)


    for i in project_serialize.data:
        # i['model_type_name'] = Moudeltype.objects.filter(model_type=i['model_type']).first().model_type_name
        mt = Moudeltype.objects.filter(model_type=i['model_type']).first()
        i['model_type_name'] = mt.model_type_name if mt else "未知类型"

    res = {}
    res['code'] = 20000
    res['msg'] = ''
    res['data'] = {'pageNum': int(page), 'pageSize': int(pageSize), 'total': project_list.count(), 'records': project_serialize.data}
    # print('data',res['data'])
    return Response(res)



@api_view(["GET"])
def get_project(request):
    project_list = Project.objects.exclude(id='-1')#.all()
    projects = ProjectSerializer(project_list, many=True)
    data = []
    for project in projects.data:
        data.append({ 'value': project['id'], 'label': project['project_name'] })
    return Response({'code': 200, 'data': data })





@api_view(["GET"])
def get_project_list_modeltype(request):
    # print(get_client_ip(request),dict(request.headers))
    project_list = Moudeltype.objects.all()#.exclude(id='-1')  # .all()#.filter(id!=-1)
    if request.GET.get('model_type_name'):
        project_list = project_list.filter(model_type_name=request.GET.get('model_type_name'))#.all()#.filter(id!=-1)

    projects = MoudeltypeSerializer(project_list, many=True)
    data = []
    temp=[]
    for project in projects.data:
        res=Moudeltype.objects.get(model_type=project['model_type']).model_type_name
        if res in temp:
            continue
        temp.append( res)
        data.append({ 'value': project['model_type'], 'label': res })
    print('check:::',len(data),data)
    return Response({'code': 200, 'data': data })



@api_view(["PUT"])
def project_update(request):
    try:
        new_data = request.query_params
        project_id = new_data['id']
        new_project_name = new_data['project_name']
        
        # 检查名称是否与其他项目重复（排除当前项目）
        existing_project = Project.objects.filter(
            project_name=new_project_name
        ).exclude(id=project_id).first()
        
        if existing_project:
            return Response({
                'code': 400, 
                'message': f'图像名称"{new_project_name}"已存在，请使用其他名称'
            }, status=400)
        
        project = Project.objects.get(id=project_id)
        image = json.loads(new_data['picture'])['picture']
        
        # 准备更新字段
        update_fields = {
            'last_mod_time': timezone.now(),
            'project_name': new_project_name,
            'project_desc': new_data['project_desc'],
        }
        
        # 处理 model_type 外键（如果存在）
        if 'model_type' in new_data:
            model_type_id = new_data['model_type']
            try:
                # 验证模型类型是否存在，并同步更新 model_type_name
                model_type_obj = Moudeltype.objects.get(model_type=model_type_id)
                update_fields['model_type_id'] = model_type_id
                update_fields['model_type_name'] = model_type_obj.model_type_name
            except Moudeltype.DoesNotExist:
                return Response({
                    'code': 404, 
                    'message': f'模型类型ID {model_type_id} 不存在'
                }, status=404)
        
        # 处理 model_type_name 字段
        # 如果是数字，说明前端传的是 model_type 的ID，需要同步更新两个字段
        if 'model_type_name' in new_data:
            model_type_name_value = new_data['model_type_name']
            
            # 检查是否为纯数字（说明是ID）
            if isinstance(model_type_name_value, str) and model_type_name_value.isdigit():
                try:
                    # 按ID查询模型类型
                    model_type_id = int(model_type_name_value)
                    model_type_obj = Moudeltype.objects.get(model_type=model_type_id)
                    # 同步更新外键和名称
                    update_fields['model_type_id'] = model_type_id
                    update_fields['model_type_name'] = model_type_obj.model_type_name
                except Moudeltype.DoesNotExist:
                    return Response({
                        'code': 404, 
                        'message': f'模型类型ID {model_type_name_value} 不存在'
                    }, status=404)
            else:
                # 如果不是数字，直接更新 model_type_name（允许自定义名称）
                update_fields['model_type_name'] = model_type_name_value
        
        # 添加 is_trainset 字段（如果存在）
        if 'is_trainset' in new_data:
            update_fields['is_trainset'] = int(new_data['is_trainset'])
        
        # 添加 is_example 字段（如果存在）
        if 'is_example' in new_data:
            update_fields['is_example'] = int(new_data['is_example'])
        
        if not image:
            project = Project.objects.filter(id=project_id)
            project.update(**update_fields)
        else:
            # image = urllib.request.unquote(image[7:])
            project = Project.objects.filter(id=project_id)
            project.update(**update_fields)

            picture = Picture.objects.get(picture=image[7:])
            temp = Picture.objects.filter(project=Project.objects.get(id=project_id))
            if temp:
                temp = temp.first().delete()
            picture.picture_name = picture.picture.name
            picture.project=Project.objects.get(id=project_id)
            picture.save()

        return Response({'code': 200, 'message': '模板修改成功！'})
        
    except Project.DoesNotExist:
        return Response({'code': 404, 'message': '项目不存在'}, status=404)
    except Picture.DoesNotExist:
        return Response({'code': 404, 'message': '图片不存在'}, status=404)
    except Moudeltype.DoesNotExist:
        return Response({'code': 404, 'message': '模型类型不存在'}, status=404)
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print('更新项目时发生错误:', error_msg)
        return Response({'code': 500, 'message': f'更新失败: {str(e)}'}, status=500)

# @api_view(["PUT"])
# def project_update_moudel(request):

#     try:
#         new_data = request.data  # PUT 请求中使用 request.data
#         model_type = new_data.get('model_type')
#         model_type_name = new_data.get('model_type_name')
#         picture_json = new_data.get('picture')

#         if not (model_type and model_type_name and picture_json):
#             return Response({'code': 400, 'message': '缺少必要参数'})

#         project = Moudeltype.objects.get(model_type=model_type)

#         # 解析图片名
#         picture_data = json.loads(request.query_params['picture'])['picture'][7:]
#         new_image_path = picture_data
#         if not new_image_path:
#             return Response({'code': 400, 'message': '图片路径格式错误'}, status=status.HTTP_400_BAD_REQUEST)

#         # 去除路径前缀，例如 'media/xxx.jpg' 去掉前缀
#         new_image_name = new_image_path.split('/')[-1]

#         # 当前图片是否存在并一致
#         current_image_name = getattr(getattr(project.picture, 'picture_name', ''), 'strip', lambda: '')()
#         if current_image_name != new_image_name:
#             # 删除旧图
#             if project.picture:
#                 project.picture.delete()

#             # 关联新图片
#             try:
#                 picture = Picture.objects.get(picture__icontains=new_image_name)
#                 picture.picture_name = picture.picture.name
#                 picture.moudeltype = project
#                 picture.save()
#                 project.picture = picture
#             except Picture.DoesNotExist:
#                 print('图片未找到')
#                 return Response({'code': 404, 'message': '图片未找到'}, status=status.HTTP_404_NOT_FOUND)

#         # 修改文字信息
#         project.model_type_name = model_type_name
#         project.save()

#         return Response({'code': 200, 'message': '模板修改成功！'})

#     except Moudeltype.DoesNotExist:
#         print('模型类型不存在')
#         return Response({'code': 404, 'message': '模型类型不存在'}, status=status.HTTP_404_NOT_FOUND)
#     except Exception as e:
#         import traceback
#         print('服务器错误',traceback.format_exc())
#         return Response({'code': 500, 'message': f'服务器错误: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
@api_view(["PUT"])
def project_update_moudel(request):
    new_data = request.query_params
    project = Moudeltype.objects.get(model_type=new_data['model_type'])
    
    try:
        # 解析新图片路径
        picture_data = json.loads(new_data['picture'])
        new_image = picture_data['picture']
        if new_image:
            new_image = new_image[7:]  # 移除 '/media/' 前缀
        
        # 获取当前项目的图片（如果存在）
        current_picture = None
        try:
            current_picture = project.picture
            current_image_name = current_picture.picture_name if current_picture else None
        except Picture.DoesNotExist:
            # 项目当前没有关联图片
            current_picture = None
            current_image_name = None
        
        # 情况1：新图片为空（不上传图片或删除图片）
        if not new_image:
            # 如果原来有图片，删除关联
            if current_picture:
                current_picture.moudeltype = None
                current_picture.save()
            # 只更新模型名称
            project.model_type_name = new_data['model_type_name']
            project.save()
            
        # 情况2：有新图片
        else:
            # 检查图片是否与当前相同
            if current_picture and current_image_name == new_image:
                # 图片没变，只更新模型名称
                project.model_type_name = new_data['model_type_name']
                project.save()
            else:
                # 图片有变化，需要更新图片关联
                try:
                    # 查找新图片
                    new_picture = Picture.objects.get(picture=new_image)
                    
                    # 如果原来有图片，先解除关联
                    if current_picture:
                        current_picture.moudeltype = None
                        current_picture.save()
                    
                    # 关联新图片
                    new_picture.moudeltype = project
                    new_picture.picture_name = new_picture.picture.name
                    new_picture.save()
                    
                    # 更新模型名称
                    project.model_type_name = new_data['model_type_name']
                    project.save()
                    
                except Picture.DoesNotExist:
                    # 新图片不存在，只更新模型名称
                    project.model_type_name = new_data['model_type_name']
                    project.save()
                    return Response({'code': 400, 'message': '指定的图片不存在，仅更新了模型名称！'})
        
        return Response({'code': 200, 'message': '模板修改成功！'})
        
    except Exception as e:
        import traceback
        print('更新模板时发生错误:', traceback.format_exc())
        return Response({'code': 500, 'message': f'更新模板失败: {str(e)}'}, status=500)



@api_view(["delete"])
def project_delete(request):
    print('????')
    id = request.query_params['id']
    print('idid',id)
    project = Project.objects.get(id=id)
    print('project',project)
    try:
        project.picture.picture.delete()
    except Exception as e:
        print("删除图片失败", e)
    project.delete() 
    return Response({'code': 200, 'data': 'success'})


@api_view(['POST'])
def create_mark(request):
    data = request.query_params
    mark_form = MarkForm(data)
    if mark_form.is_valid():
        try:
            mark = mark_form.save(False)
            mark.moudeltype_id = data['moudeltype']
            mark.save(True)
            return Response({'code': 200, 'message': '添加mark成功！'})
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return Response({'code': 500, 'message': '插入值失败！'}, status=500)
    else:
        # if ['label已存在'] in mark_form.errors.values():
        #     return  Response({'code': 500, 'message': 'Label已存在！'})
        
        return Response({'code': 400, 'message': '插入值失败！'}, status=400)


@api_view(['GET'])
def get_mark_list(request):
    page = request.GET.get('pageNum')
    pageSize = request.GET.get('pageSize')
    mark_list = Mark.objects.exclude(id=-1)#.all()
    if request.GET.get('moudeltype'):
        mark_list = mark_list.filter(moudeltype_id=request.GET.get('moudeltype'))
    if request.GET.get('project'):
        mark_list = mark_list.filter(project=request.GET.get('project'))
    if request.GET.get('mark_name'):
        mark_list = mark_list.filter(mark_name__contains=request.GET.get('mark_name'))

    # 创建分页对象
    ptr = Paginator(mark_list, pageSize)
    # 分页
    masters = ptr.page(page)
    # 序列化
    mark_serialize = MarkSerializer(masters, many=True)

    res = {}
    res['code'] = 20000
    res['msg'] = ''
    for i in mark_serialize.data:
        try:
            i['model_type_name']=Project.objects.filter(id=i['project']).first().model_type.model_type_name
        except:
            print(i['project'])
    res['data'] = {'pageNum': int(page), 'pageSize': int(pageSize), 'total': mark_list.count(),
                   'records': mark_serialize.data}
    return Response(res)



@api_view(['GET'])
def get_mark_list_train(request):
    if request.GET.get('pageNum'):
        page = request.GET.get('pageNum')
    else:
        page = 1
    if request.GET.get('pageSize'):
        pageSize = request.GET.get('pageSize')
    else:
        pageSize = 500
    mname=request.GET.get('model_name')
    mark_list = Markresults.objects.all()
    if mname:
        mark_list=mark_list.filter(Moudeltype_id=mname)
    if request.GET.get('mark_name'):
        mark_list = mark_list.filter(mark_id=request.GET.get('mark_name'))
    if request.GET.get('is_trainset'):
        mark_list = mark_list.filter(is_trainset=request.GET.get('is_trainset'))
    if request.GET.get('project_name'):
        mark_list = mark_list.filter(project_id=request.GET.get('project_name'))
    ptr = Paginator(mark_list, pageSize)
    # 分页
    masters = ptr.page(page)
    # 序列化
    mark_serialize = MarkresultsSerializer(masters, many=True)
    res = {}
    res['code'] = 20000
    res['msg'] = ''
    ans=[]

    for i in mark_serialize.data:

        if len(Mark.objects.filter(id=i['mark']).values())==0:
            print(i)
        i['mark_name'] = Mark.objects.get(id=i['mark']).mark_name
        i['project_name'] = Project.objects.filter(id=i['project']).first().project_name
        # i['is_trainset'] = Project.objects.filter(id=i['project']).first().is_trainset
        # Use Moudeltype from Markresults, not Project
        if i.get('Moudeltype'):
            mtype = Moudeltype.objects.filter(model_type=i['Moudeltype']).first()
            i['model_name'] = mtype.model_type_name if mtype else ''
        else:
            i['model_name'] = ''
        temp=Picture.objects.filter(project=i['project']).first()
        if temp:
            i['picture'] = Picture.objects.filter(project=i['project']).first().picture.url
            pt=os.path.dirname(os .path.abspath(__file__))+i['picture']
            pt= unquote(pt)
            pt=re.sub('TemplateOCR','static',pt)
            
            img=cv2.imread(pt)
            box = transPoint(i['points'])
            img=draw_boxes(img, [box])
            retval, buffer = cv2.imencode('.jpg', img)
            image = base64.b64encode(buffer)
            i['picture']=image
        else:
            i['picture'] = ''




    res['data'] = {'pageNum': int(page), 'pageSize': int(pageSize), 'total': mark_list.count(),
                   'records': mark_serialize.data}
    return Response(res)








@api_view(['GET'])
def get_mark_list1(request):
    if request.GET.get('pageNum'):
        page = request.GET.get('pageNum')
    else:
        page=1
    if request.GET.get('pageSize'):
        pageSize = request.GET.get('pageSize')
    else:
        pageSize=500
    mark_list = Mark.objects.all()
    if request.GET.get('mark_name'):
        mark_list = Mark.objects.filter(mark_name=request.GET.get('mark_name'))
    ptr = Paginator(mark_list, pageSize)
    # 分页
    masters = ptr.page(page)
    # 序列化
    mark_serialize = MarkSerializer(masters, many=True)
    res = {}
    res['code'] = 20000
    res['msg'] = ''
    res['data'] = {'pageNum': int(page), 'pageSize': int(pageSize), 'total': mark_list.count(),
                   'records':  mark_serialize.data}
    return Response(res)


# @api_view(["PUT"])
# def mark_update(request):
#     new_data = request.query_params.copy()

#     mark_form = MarkForm(new_data)
#     if mark_form.is_valid():
#         mark = mark_form.save(False)
#         mark.id = new_data['id']
#         mark.save()

#         return Response({'code': 200, 'message': '修改成功！'})
#     return Response({'code': 403, 'message': '请确认数据输入正确'},status=403)
@api_view(["PUT"])
def mark_update(request):
    try:
        # 兼容 query string 和 body
        data = request.data.copy()
        for k, v in request.query_params.items():
            if k not in data:
                data[k] = v

        mark_id = data.get("id")
        if not mark_id:
            return Response({"code": 400, "message": "缺少 id 参数"}, status=400)

        try:
            mark = Mark.objects.get(id=mark_id)
        except Mark.DoesNotExist:
            return Response({"code": 404, "message": f"对象不存在: id={mark_id}"}, status=404)

        # 允许更新的字段
        allowed_fields = [
            "mark_name", "points", "flag", "mark_type",
            "project", "moudeltype"
        ]

        for field in allowed_fields:
            if field in data:
                # 外键要特别处理
                if field == "project" and data[field]:
                    from .models import Project
                    try:
                        mark.project = Project.objects.get(id=data[field])
                    except Project.DoesNotExist:
                        return Response({"code": 400, "message": f"无效的 project id={data[field]}"}, status=400)

                elif field == "moudeltype" and data[field]:
                    from .models import Moudeltype
                    try:
                        mark.moudeltype = Moudeltype.objects.get(model_type=data[field])
                    except Moudeltype.DoesNotExist:
                        return Response({"code": 400, "message": f"无效的 moudeltype id={data[field]}"}, status=400)

                elif field == "flag":
                    mark.flag = str(data[field]).lower() in ["true", "1", "yes"]

                elif field == "points":
                    import json
                    try:
                        mark.points = json.loads(data[field]) if isinstance(data[field], str) else data[field]
                    except Exception:
                        return Response({"code": 400, "message": "points 字段必须是 JSON 格式"}, status=400)

                else:
                    setattr(mark, field, data[field])

        mark.last_mod_time = timezone.now()
        mark.save()

        return Response({
            "code": 200,
            "message": "修改成功！",
            "data": {
                "id": mark.id,
                "mark_name": mark.mark_name,
                "points": mark.points,
                "flag": mark.flag,
                "mark_type": mark.mark_type,
                "project": mark.project.id if mark.project else None,
                "moudeltype": mark.moudeltype.model_type if mark.moudeltype else None,
                "create_time": mark.create_time,
                "last_mod_time": mark.last_mod_time,
            }
        })

    except Exception as e:
        import traceback
        return Response({
            "code": 500,
            "message": f"服务器异常: {str(e)}",
            "traceback": traceback.format_exc()
        }, status=500)


@api_view(["PUT"])
def markresults_update(request):
    try:
        # 从查询参数获取数据
        new_data = request.query_params.copy()
        
        # 检查必需的参数
        if 'id' not in new_data:
            return Response({'code': 400, 'message': '缺少id参数'})
        
        # 检查记录是否存在
        try:
            markresult = Markresults.objects.get(id=new_data['id'])
        except Markresults.DoesNotExist:
            return Response({'code': 404, 'message': '记录不存在'})
        
        # 准备更新的字段
        update_fields = {'last_mod_time': timezone.now()}
        
        # 处理 content 字段
        if 'content' in new_data:
            update_fields['content'] = new_data['content']
        
        # 处理 is_trainset 字段
        if 'is_trainset' in new_data:
            value = new_data['is_trainset']
            # 转换为整数：支持 '0', '1', 'true', 'false' 等格式
            if isinstance(value, str):
                if value.isdigit():
                    update_fields['is_trainset'] = int(value)
                else:
                    update_fields['is_trainset'] = 1 if value.lower() in ['true', 'yes'] else 0
            else:
                update_fields['is_trainset'] = int(value)
        
        # 处理 points 字段
        if 'points' in new_data:
            value = new_data['points']
            if isinstance(value, str):
                try:
                    update_fields['points'] = json.loads(value)
                except:
                    update_fields['points'] = value
            else:
                update_fields['points'] = value
        
        # 处理 flag 字段
        if 'flag' in new_data:
            value = new_data['flag']
            update_fields['flag'] = str(value).lower() in ['true', '1', 'yes']
        
        # 处理 project 外键
        if 'project' in new_data:
            try:
                project_id = int(new_data['project'])
                # 验证Project是否存在
                Project.objects.get(id=project_id)
                update_fields['project_id'] = project_id
            except (ValueError, Project.DoesNotExist) as e:
                print(f"无效的project ID: {new_data['project']}, 错误: {e}")
        
        # 处理 mark 外键
        if 'mark' in new_data:
            try:
                mark_id = int(new_data['mark'])
                # 验证Mark是否存在
                Mark.objects.get(id=mark_id)
                update_fields['mark_id'] = mark_id
            except (ValueError, Mark.DoesNotExist) as e:
                print(f"无效的mark ID: {new_data['mark']}, 错误: {e}")
        
        # 统一处理模型类型外键，支持 model_name、Moudeltype、moudeltype 三种参数
        moudeltype_id_final = None
        moudeltype_error = None
        print(f"[DEBUG] 入参: {new_data}")
        # 优先级：model_name > Moudeltype > moudeltype
        if 'model_name' in new_data:
            model_name_value = new_data['model_name']
            print(f"[DEBUG] model_name参数: {model_name_value}")
            if isinstance(model_name_value, str) and model_name_value.isdigit():
                try:
                    moudeltype_id_final = int(model_name_value)
                    print(f"[DEBUG] 作为ID查找Moudeltype: {moudeltype_id_final}")
                    Moudeltype.objects.get(model_type=moudeltype_id_final)
                except (ValueError, Moudeltype.DoesNotExist) as e:
                    moudeltype_error = f"无效的model_name ID: {model_name_value}, 错误: {e}"
            else:
                try:
                    print(f"[DEBUG] 作为名称查找Moudeltype: {model_name_value}")
                    moudeltype_obj = Moudeltype.objects.get(model_type_name=model_name_value)
                    moudeltype_id_final = moudeltype_obj.model_type
                except Moudeltype.DoesNotExist as e:
                    moudeltype_error = f"无效的model_name: {model_name_value}, 错误: {e}"
        elif 'Moudeltype' in new_data:
            try:
                moudeltype_id_final = int(new_data['Moudeltype'])
                print(f"[DEBUG] Moudeltype参数: {moudeltype_id_final}")
                Moudeltype.objects.get(model_type=moudeltype_id_final)
            except (ValueError, Moudeltype.DoesNotExist) as e:
                moudeltype_error = f"无效的Moudeltype ID: {new_data['Moudeltype']}, 错误: {e}"
        elif 'moudeltype' in new_data:
            try:
                moudeltype_id_final = int(new_data['moudeltype'])
                print(f"[DEBUG] moudeltype参数: {moudeltype_id_final}")
                Moudeltype.objects.get(model_type=moudeltype_id_final)
            except (ValueError, Moudeltype.DoesNotExist) as e:
                moudeltype_error = f"无效的moudeltype ID: {new_data['moudeltype']}, 错误: {e}"
        if moudeltype_error:
            print(f"[DEBUG] moudeltype_error: {moudeltype_error}")
            return Response({'code': 400, 'message': moudeltype_error, 'debug': {'params': new_data}}, status=400)
        if moudeltype_id_final is not None:
            update_fields['Moudeltype_id'] = moudeltype_id_final
        print(f"[DEBUG] update_fields 最终: {update_fields}")
        
        # 处理 sub_id 字段
        if 'sub_id' in new_data:
            update_fields['sub_id'] = str(new_data['sub_id'])
        
        # 用 save 方法更新，保证外键赋值生效
        markresult = Markresults.objects.get(id=new_data['id'])
        for k, v in update_fields.items():
            print(f"[DEBUG] setattr: {k} = {v}")
            setattr(markresult, k, v)
        markresult.save()
        print(f"[DEBUG] 数据库实际 Moudeltype_id: {markresult.Moudeltype_id}")
        print(f"成功更新记录 ID: {new_data['id']}, 更新字段: {update_fields}")
        return Response({'code': 200, 'message': '修改成功！'})
        
    except Exception as e:
        import traceback
        print(f"更新失败: {str(e)}")
        print(traceback.format_exc())
        return Response({'code': 500, 'message': f'更新失败: {str(e)}'})



@api_view(["POST"])
def add_points(request):
    data = request.data.copy()  # 确保是可变字典
    print('[input data]', data)
    import datetime
    if 'project' in data:
        # 新增分支，剔除id，防止主键冲突
        data_for_create = data.copy()
        # 新增时彻底剔除id，防止主键NOT NULL错误
        if 'id' in data_for_create:
            data_for_create.pop('id')
        modeltypeid = Project.objects.get(id=data['project']).model_type.model_type
        val = Markresults.objects.all().values()
        for i in val:
            if json.loads(data['points']) == i['points']:
                Markresults.objects.filter(id=i['id']).update(
                    mark_id=data['id'],
                    content=data['content'],
                    sub_id=data['sub_id'],
                    last_mod_time=datetime.datetime.now()
                )
                return Response({'code': 200, 'status': 'success', 'message': 'label更新'})
        try:
            now = datetime.datetime.now()
            # 直接用Django ORM插入，所有参数都从data_for_create取，id绝不传入
            from django.db.models import Max

            # 手动生成新的主键 id（兼容非自增主键表）
            max_id = Markresults.objects.aggregate(Max('id'))['id__max'] or 0
            new_id = max_id + 1

            Markresults.objects.create(
                id=new_id,  # ✅ 手动指定主键
                points=json.loads(data_for_create['points']) if isinstance(data_for_create['points'], str) else data_for_create['points'],
                create_time=now,
                last_mod_time=now,
                project_id=data_for_create['project'],
                mark_id=data['id'],
                flag=True,
                Moudeltype_id=modeltypeid,
                is_trainset=1,
                content=data_for_create['content'],
                sub_id=data_for_create['sub_id'].split('-')[-1]
            )
            return Response({'code': 200, 'status': 'success', 'message': '添加成功！'})
        except Exception as e:
            print('[错误信息]',e)
            print('添加points失败', e)
            return Response({'code': 500, 'status': 'error', 'message': '请确认数据输入正确' + str(e)}, status=500)
    else:
        if isinstance(data['points'], str):
            Markresults.objects.filter(id=data['id']).update(points=json.loads(data['points']), flag=False, last_mod_time=datetime.datetime.now())
        else:
            Markresults.objects.filter(id=data['id']).update(points=data['points'], flag=False, last_mod_time=datetime.datetime.now())
        return Response({'code': 200, 'status': 'success', 'message': '更新成功！'})

@api_view(['delete'])
def markresults_delete(request):
    mark = Markresults.objects.get(id=request.query_params['id'])
    # mark = Mark.objects.get(id=request.query_params['id'])
    mark.delete()
    return Response({'code': 200, 'message': '删除成功'})

@api_view(['DELETE'])
def mark_delete(request):
    mark_id = request.query_params.get('id')

    # 删除 Mark
    try:
        mark = Mark.objects.get(id=mark_id)
        mark.delete()
    except Mark.DoesNotExist:
        return Response({'code': 404, 'message': 'Mark不存在'}, status=404)

    # 删除关联的 Markresults（如果有）
    markres = Markresults.objects.filter(mark_id=mark_id).first()
    if markres:
        markres.delete()

    return Response({'code': 200, 'message': '删除成功'})


@api_view(['GET'])
def get_img_by_project_id(request):
    tem={}
    if len(request.query_params)==0 or 'project_id' not in request.query_params:
        tem['picture'] = '/media/template_img/img/defaultPic.png'
        return Response({'code': 200, 'data': tem})
    if request.query_params['project_id']=='None':
        tem['picture'] = '/media/template_img/img/defaultPic.png'
        return Response({'code': 200, 'data': tem})
    project_id = request.query_params['project_id']
    if 'tag' in request.query_params and request.query_params['tag']=='1':
        print(Moudeltype.objects.all())
        img = Moudeltype.objects.get(model_type=project_id).picture
    else:
        img = Project.objects.get(id=project_id).picture
    picture = PictureSerializer(img)

    return Response({'code': 200, 'data': picture.data})


@api_view(['GET'])
def get_mark_by_project_id(request):
    project_id = request.query_params['project_id']
    mark = Project.objects.get(id=project_id).mark_set.filter().values()
    mark_serializer = MarkSerializer(mark, many=True)

    return Response({'code': 200, 'data': mark_serializer.data})


@api_view(['GET'])
def get_mark_by_Moudeltype_id(request):
    project_id = request.query_params['project_id']
    type = Project.objects.get(id=project_id).model_type_id
    mark = Mark.objects.filter(moudeltype_id=type).values()
    mark_serializer = MarkSerializer(mark, many=True)

    return Response({'code': 200, 'data': mark_serializer.data})

@api_view(['GET'])
def get_mark_points(request):
    project_id = request.query_params['project_id']

    markres=MarkresultsSerializer(Markresults.objects.filter(project_id=project_id), many=True).data

    for i in markres:
        i['mark_name']=Mark.objects.filter(id=i['mark']).first().mark_name
        # print(Mark.objects.filter(id=i['mark']).first().mark_name)
        i['points']=json.dumps( i['points'] ,ensure_ascii=False)


    return Response({'code': 200, 'data': markres})
# @api_view(['GET'])
# def get_mark_by_project_id(request):
#     project_id = request.query_params['project_id']
#     mark = Project.objects.get(id=project_id).mark_set.filter().values()
#     # print(mark)
#     mark_serializer = MarkSerializer(mark, many=True)
#     # print(mark_serializer.data)
#     return Response({'code': 200, 'data': mark_serializer.data})
#
# @api_view(['GET'])
# def get_mark_points(request):
#     project_id = request.query_params['project_id']
#     mark = Project.objects.get(id=project_id).mark_set.filter(flag=True).values()
#     mark_serializer = MarkSerializer(mark, many=True)
#     # print(mark_serializer.data)
#     return Response({'code': 200, 'data': mark_serializer.data})


@api_view(['DELETE'])
def del_point_by_mark_id(request):
    mark_id = request.query_params['id']
    if len(str(mark_id))>15:
        Markresults.objects.filter(sub_id=mark_id.split('-')[-1]).delete()
    else:
        Markresults.objects.filter(id=mark_id).delete()
    return Response({'code': 200, 'message': 'success'})


@api_view(['GET'])
def build_model_url(request, project_id):

    project = Project.objects.get(id=project_id)
    img = PictureSerializer(Picture.objects.get(project=project))
    ret = ProjectWithMark(project)
    data = recognize(img.data, ret.data)
    return Response(data)

@api_view(['POST'])
def api(request, project_id):
    project = Project.objects.get(id=project_id)
    file = request.data.get('file')
    file_bytes = file.read()
    img = Image.open(BytesIO(file_bytes))
    img = rotate(img)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    ret = ProjectWithMark(project)
    data, result_img = recognize(img, ret.data)

    retval, buffer = cv2.imencode('.jpg', result_img)
    image = base64.b64encode(buffer)
    return Response({'code': 200, 'message': 'success', 'success': True, 'data': data, 'image': image})


def rotate(img):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(img._getexif().items())

        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)
    except Exception as e:
        return img
    return img

def specialcase_法定代表人身份证明书(ans):
    text_combine=''
    for i in ans:
        text_combine=text_combine+re.sub('\n','',i['A'])
    temp=[i for  i in re.split('法定代表人身份证明书',text_combine) if i!='']
    temp=re.split('同志',temp[0])
    name=temp[0]
    temp=[i for i in re.split('[担任]|职务[,，（(]',temp[1]) if i!='']
    position=temp[1]
    temp=[i for i in re.split('身份证号码[：:]|联系电话[：:]|[)）]',temp[2]) if i!='']
    idnum=temp[0]
    phonenum=temp[1]
    company=[i for i in re.split('[.。0-9年月日]', temp[-1])if i!=''][-1]
    ans = [
        ans[0],
        {'Q': '姓名', 'A': name},
        {'Q': '职位', 'A': position},
        {'Q': '身份证号码', 'A': idnum},
        {'Q': '电话号码', 'A': phonenum},
        {'Q': '公司名称', 'A': company}
    ]
    return ans



def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip




@api_view(['POST']) 
def show_data(request):
    try:
        project_id = request.data.get('project_id')
        savename = request.data.get('save_name')
        file = request.data.get('file')
        filename = file.name
        tagg=0
        if not savename:
            tagg=1
            savename = filename
        path1 = os.path.dirname(os.path.abspath(__file__))

        if '.pdf' ==filename[-4:]:
            
            save_path_temp=os.path.dirname(path1)+'/static/media/template_pdf/'+filename 
            
            with open(save_path_temp,'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
            img = convert_pdf_with_adaptive_size(save_path_temp)[0]
            clear_directory(os.path.dirname(path1)+'/static/media/template_pdf/')
            filename = filename +'.jpeg'
        else:

            file_bytes = file.read()
            img = Image.open(BytesIO(file_bytes))

            save_path_temp=os.path.dirname(path1)+r'/static/media/template_img/'+filename 
        img = rotate(img)

        
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        res_img=resize_if_needed(img,1500)
        img_ = redink_remover(res_img)
        if '营业执照' in savename:
            print('[***********]')
            r, _ = text_detector(img_)
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(use_onnx=True, use_angle_cls=True, lang="ch", det_db_unclip_ratio=1.0,
                            cls_model_dir='./Funcs/inferenced_models/cls.onnx',
                            det_model_dir='./Funcs/inferenced_models/det_ct.onnx',
                            rec_model_dir='./Funcs/inferenced_models/rec_v4.onnx',
                            use_dilation=True)
            print('【ocr loaded】')
            angles = []
            for sub in range(2,len(r)//2):
                subbox = r[sub]
                box = subbox
                tl = box[0]
                tr = box[1]
                br = box[2]
                bl = box[3]
                tag1 = max(int(tr[1]), int(bl[1])) - max(min(int(tr[1]), int(bl[1])), 0)
                tag2 = max(int(tl[0]), int(br[0])) - max(min(int(tl[0]), int(br[0])) - 5, 0)
                crop = img_[max(min(int(tr[1]), int(bl[1])) - 5, 0):max(int(tr[1]), int(bl[1])) + 5,
                       max(min(int(tl[0]), int(br[0])) - 5, 0):max(int(tl[0]), int(br[0])) + 5]

                result = ocr.ocr(crop, cls=True, rec=False, det=False)
                if tag1 < tag2 and result[0][0][0] == '0':
                    result = 0
                else:
                    result = 1
                # if tag1 > tag2:
                #     result = 1
                # else:
                #
                #     result = 0
                angles.append(result)
            if np.mean(angles)>0.9:
                img_=rotateImage(img_,90)
                r, _ = text_detector(img_)
            angles = [] 
            for sub in range(2,len(r)//2):
                subbox = r[sub]
                box = subbox
                tl = box[0]
                tr = box[1]
                br = box[2]
                bl = box[3]
                crop = img_[max(min(int(tr[1]), int(bl[1])) - 5, 0):max(int(tr[1]), int(bl[1])) + 5,
                       max(min(int(tl[0]), int(br[0])) - 5, 0):max(int(tl[0]), int(br[0])) + 5]

                result = ocr.ocr(crop, cls=True, rec=False, det=False)  # ,rec=False
                angles.append(int(result[0][0][0]))

            if np.mean(angles)>60:
                img_=rotateImage(img_, 180)

        
        # --------Edit in 2025/06/05 -------startflag---------
        # print('[1filenamefilename]',filename)
        # print(os.path.dirname(path1)+r'/static/media/template_img/'+filename)
        # if tagg!=1:
        #     cv2.imwrite(os.path.dirname(path1)+r'/static/media/template_img/'+filename,img_)
        # res,res_img,savedpath=text_pred_res_det( img_, save_path_temp,savename,SerPredictor)
        res, res_img, ans = SER_generator(img_)#(img,text_sys,kie_predictor)
        cv2.imwrite(os.path.dirname(path1)+r'/static/media/template_img/temp.jpg',img_)
        # print('\n[DEBUG   resresres]',res)
        retval, buffer = cv2.imencode('.jpg', res_img)
        image = base64.b64encode(buffer)

        retval, buffer = cv2.imencode('.jpg', img_)
        img_ = base64.b64encode(buffer)

        # print(res)  

        if tagg==1:
            return Response({'code': 200, 'message': 'success', 'success': True, 'data': ans},status=200)
        return Response({'code': 200, 'message': 'success', 'success': True, 'data': ans, 'image': image,'savedpath':os.path.dirname(path1)+r'/static/media/template_img/'+filename,'oriimgpath':file.name,'origindata':json.dumps( res ,ensure_ascii=False),'preprocessedimage':img_,'imagename':filename}
        ) #,'preprocessedimage':img_
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        # --------Edit in 2025/06/05 -------endflag--------
        return Response({'code': 501, 'message':str(traceback.format_exc()),'data':{}},status=501)

 

# 上传至DB
@api_view(["POST"])
def uploadData2DB(data): #items: List[Item]
    
    data_=json.loads(data.POST['content'])
    project_id=data.POST['project_id']
    create_time = data.POST['create_date']
    project_name = data.POST['project_name']
    ori_data = data.POST['origin_res_data']
    ori_data=json.loads(ori_data)
    modeltype_name=data.POST['modeltype_name']
    imagename=data.POST['imagename']


    projid=Project.objects.all().aggregate(Max('id'))['id__max']
    if projid==None:
        projid=-1
    projid=projid+1



    if ori_data==[]:
        return Response({'code':400,'message': '数据为空!', 'success': False}, status=400)

    badtag=-1
    id1=-100
    id2=-100

    print({'id': projid, 'project_name': project_name,
                       'project_desc': "aoto generated when " + create_time, 'is_trainset': 0,
                       'model_type': project_id, 'model_type_name': modeltype_name, 'is_example': 0,
                       'model_type_id': int(project_id)})
    

    try:
        print('Creating new project entry...')
        label2write = Project()
        label2write.project_desc="aoto generated when " + create_time
        label2write.project_name=project_name
        label2write.is_trainset=0
        label2write.model_type=Moudeltype.objects.get(model_type=project_id)
        label2write.model_type_name=modeltype_name
        label2write.is_example=0
        label2write.model_type_id=int(project_id)
        label2write.save()
        print('created project with name:', project_name)

    except Exception as e:
        print(e)
        return Response({'code': 500, 'message': '插入数据库失败', 'success': False}, status=500)
    print('creating picture entry...    ')
    projid=Project.objects.all().aggregate(Max('id'))['id__max']
    
    image = data.FILES.get('file')
    if image==None:
        image=data.POST['file']
        image=base64.b64decode(image)
        # print('aaa',image)
    picture = Picture()
    pic_io = BytesIO(image)
    img = Image.open(pic_io )

    img_rotate = rotate(img)
    img_rotate.save(pic_io, img.format)
    content_type=''
    if 'jpeg' in imagename[-5:] or 'jpg' in imagename[-4:]:
        content_type='image/jpeg'
    else:
        content_type='image/png'
    print('\n[imagename]\n',imagename)
    pic = InMemoryUploadedFile(
        file=pic_io,
        field_name=None,
        name='template_img/res_' + imagename,
        content_type=content_type,
        size=img_rotate.size,
        charset=None
    )

    picture.picture = pic
    picture.picture_name = picture.picture.name
    picture.project = Project.objects.get(id=projid)
    picture.save()
    print('picture entry created.')
    



    try:
        print('Inserting markresults...')
        for item in ori_data:

            if item['pred']=="NONE":
                item['pred']='None'
            if item['pred']=='other' or item['pred']=='others' or item['pred']=='O' or item['pred']=='o':
                item['pred']='others'

            # 原有逻辑尝试获取 mark id（兼容历史逻辑）
            temp = None
            try:
                temp = getMarkid(item['pred'], item['points'], create_time, create_time, projid, project_id)
            except Exception:
                temp = None

            # 如果 getMarkid 返回了 id，先检查该 id 是否存在
            mark_obj = None
            if temp is not None:
                try:
                    mark_obj = Mark.objects.filter(id=temp).first()
                except Exception:
                    mark_obj = None

            # 如果没有找到，通过 (mark_name, project) 再查一次
            if not mark_obj:
                try:
                    mark_obj = Mark.objects.filter(mark_name=item['pred'], project_id=projid).first()
                except Exception:
                    mark_obj = None

            # 如果仍然没有，创建一个新的 Mark 条目
            if not mark_obj:
                mark_obj = Mark.objects.create(
                    mark_name=item['pred'],
                    points={},
                    create_time=timezone.now(),
                    last_mod_time=timezone.now(),
                    project_id=projid,
                    flag=True,
                    mark_type=item.get('mark_type','TX'),
                    moudeltype_id=project_id
                )
                print('Created new Mark:', mark_obj.id, mark_obj.mark_name)

            # 最终使用 mark_obj.id 作为外键
            temp = mark_obj.id
            print('temp (mark id):', temp)

            markresid = Markresults.objects.all().aggregate(Max('id'))['id__max']
            if markresid==None:
                markresid=-1
            markresid=markresid+1
            markres_dict={'id':markresid, 'create_time':timezone.now(), 'last_mod_time':timezone.now(),
                          'points':json.dumps( point_trans(item['points'])),
                          'project_id':projid,'mark_id':temp,'flag':1,
                          'Moudeltype_id':project_id, 'is_trainset':1,'content':item.get('transcription',''),'sub_id':markresid}
            proj_from1 = MarkresultsForm(markres_dict)
            # print(proj_from1.is_valid())
            if proj_from1.is_valid():


                proj_inser = proj_from1.save(False)
                proj_inser.project_id=projid
                proj_inser.id = markresid
                proj_inser.mark_id=temp
                proj_inser.Moudeltype_id = project_id
                proj_inser.save(True)
            else:
                badtag=1
                print('badtag')
                break
        print("done")
        if badtag==1:
            lala=lala+2
        return Response({'code': 200, 'message': '插入数据库成功', 'success': True})
    except Exception as e:
        # conn.close()
        print(e)

        return Response({'code': 500, 'message': '插入数据库失败', 'success': False}, status=500)

    return Response({'code': 500, 'message': '插入数据库失败', 'success': True}, status=500)


@api_view(["POST"])
def model_downloaddata(data):#items: List[Item]
    data_=json.loads(data.POST['content'])
    modeltype=data.POST['modelname']
    if data_==[]:
        return Response({'code':400,'message': '数据为空!', 'success': False}, status=400)
    print(data)
    path1 = os.path.dirname(os.path.abspath(__file__))
    with open (os.path.dirname(path1)+r'\static\media\template_file\\'+str(modeltype)+'.json' ,'w') as file:
        for i in data_:
            file.write(json.dumps(i,ensure_ascii=False))
        file.close()
    return Response({'code':200,'message': 'success', 'success': True, 'data':os.path.dirname(path1)+r'\static\media\template_file\\'+str(modeltype)+'.json'})



@api_view(["GET"])
def get_model_predict(request):
    page = request.GET.get('pageNum')
    pageSize = request.GET.get('pageSize')

    project_list = Moudeltype.objects.all()
    # 创建分页对象
    ptr = Paginator(project_list, pageSize)
    # 分页
    masters = ptr.page(page)
    # 序列化
    project_serialize =  ModelWithImage(masters, many=True)

    res = {}
    res['code'] = 20000
    res['msg'] = ''
    res['data'] = {'pageNum': int(page), 'pageSize': int(pageSize), 'total': project_list.count(), 'records': project_serialize.data}
    print('2check:::',len(project_serialize.data),project_serialize.data)
    return Response(res)

@api_view(["GET"])
def data_to_train(request):
    # 在训练开始前检测并设置CUDA环境
    try:
        import paddle
        print('[训练启动] 开始检测CUDA设备...')
        
        # 检测CUDA可用性
        if paddle.is_compiled_with_cuda():
            try:
                device_count = paddle.device.cuda.device_count()
                if device_count > 0:
                    print(f'[训练启动] 检测到 {device_count} 个CUDA设备')
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                else:
                    print('[训练启动] 未检测到CUDA设备，设置使用CPU')
                    os.environ['CUDA_VISIBLE_DEVICES'] = ''
            except Exception as cuda_e:
                print(f'[训练启动] CUDA设备检测异常: {str(cuda_e)}，设置使用CPU')
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
        else:
            print('[训练启动] PaddlePaddle未编译CUDA支持，设置使用CPU')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            
        # 尝试设置设备
        try:
            if os.environ.get('CUDA_VISIBLE_DEVICES') == '':
                paddle.set_device('cpu')
                print('[训练启动] 已设置使用CPU')
            else:
                paddle.set_device('gpu:0')
                print('[训练启动] 已设置使用GPU:0')
        except Exception as device_e:
            print(f'[训练启动] 设备设置失败: {str(device_e)}，强制使用CPU')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            paddle.set_device('cpu')
            
    except Exception as e:
        print(f'[训练启动] 初始化异常: {str(e)}，继续使用默认设置')
    
    update_status(0,status='preparing',progress=0)
    data = Markresults.objects.all().values()
    proj_ids = []
    for ids in data:
        if ids['project_id'] not in proj_ids:
            proj_ids.append(ids['project_id'])
    imgs = []
    for proj in proj_ids:
        img = Picture.objects.get(project_id=proj).picture.file.name
        imgs.append(img) 

    train_dct = {}
    for it in data:
      
        if it['content']==None:
            it['content']='None'
        if it['is_trainset']==0:
            continue
        temp = {'transcription': it['content'], 'points': transPoint(it['points']), 'label': Mark.objects.get(id=it['mark_id']).mark_name}
        imgpath=Picture.objects.filter(project_id=it['project_id']).first().picture.file.name
        
        if imgpath in train_dct:
            train_dct[imgpath].append(temp)
        else:
            train_dct[imgpath] = [temp]
    resfile_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r'/static/media/template_file/Label.txt'
    dataPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/data/Dataset_small'
    if os.path.exists(dataPath+'/Dataset_new'):
        clear_directory(dataPath+'/Dataset_new')
    else:
        os.mkdir(dataPath+'/Dataset_new')
    with open(resfile_path, 'w') as f:
        for i in train_dct:
            print('copy '+i+' to'+os.path.join(dataPath+'/Dataset_new',os.path.basename(i)))
            shutil.copy(i,os.path.join(dataPath+'/Dataset_new',os.path.basename(i)))
            f.write(i + '\t'+json.dumps(train_dct[i],ensure_ascii=False)+'\n')

    
    pretrainedPath_det = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))) + '/TemplateOCR/Funcs/inferenced_models/det2/selftrain/best_accuracy'
    pretrainedPath_ser= os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))) + '/TemplateOCR/Funcs/inferenced_models/ser/inference'
    ser_save_model_path= os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))) + '/TemplateOCR/output/ser_ct_tax'
    ser_save2_model_path= os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))) + '/TemplateOCR/Funcs/inferenced_models/ser/inference'
    det_saved_path=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))) + '/TemplateOCR/output/det_ct_tax/best_model/'
    det_saved2_path=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))) + '/TemplateOCR/Funcs/inferenced_models/det2'
    


    log_path=os.path.join(os.path.dirname(__file__),'train_subprocess.log')
    log_file=open(log_path,'w')
    
    # Set up CUDA environment for training
    try:
        import paddle
        device_available = paddle.device.cuda.device_count() > 0
        print(f"CUDA devices available: {device_available}")
        if not device_available:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            print("No CUDA devices detected, using CPU for training")
    except Exception as e:
        print(f"CUDA detection failed: {e}, defaulting to CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    global proc
    proc=subprocess.Popen(['python','../train_wrapper.py',pretrainedPath_det,dataPath,pretrainedPath_ser,ser_save_model_path,det_saved_path, det_saved2_path,ser_save2_model_path],
        cwd=os.path.dirname(__file__),
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
        env={**os.environ})

    
    atexit.register(cleanup)

    ret=proc.poll()
    
    # x = async_task.delay(pretrainedPath,dataPath)
    # generat_class_file()
    # DET_retrainer(pretrainedPath,dataPath)
    # print('[check path]',pretrainedPath)
    # print('[check path]',dataPath)
    return Response({'code':200, 'message':'success','data':{'state':True}})


@api_view(["GET"])
def get_training_status(request):
    task=TrainTask.objects.get(id=0) 
    if task.status=='finished':
        
        mod1=importlib.import_module("Funcs.bulkrecfuncs.modelfunc")
        mod2=importlib.import_module("task")
        importlib.reload(mod1)
        importlib.reload(mod2)
        from Funcs.bulkrecfuncs.modelfunc import SER_generator,kie_predictor ,text_sys ,convert_pdf_with_adaptive_size, clear_directory
        from task import async_task, REC_Model,DET_Model, recognizer,generat_class_file
        print('【refreshing model...】')
        text_recognizer=REC_Model()
        text_detector=DET_Model()
        reload_model()
        # SerPredictor=kie_predictor #SerPredictor_Model()
        print('【refreshing model DONE】')
        
        update_status(0,status='idle',progress=0)

    return Response({'code':200, 'data':{'state':task.status,'progress':task.progress,'error':task.error_msg}})


def training_new_model(request):
    x=async_task.delay(20,30)
    return HttpResponse('<p style="font-size:50px">调用结果'+str(x)+'</p>')


@api_view(["GET"])
def get_file_update_time(request):
    path1=os.path.dirname(os.path.abspath(__file__))
    timestamp = os.path.getmtime(path1 + '/Funcs/inferenced_models/ser/inference/model_state.pdparams')

    timestruct = localtime(timestamp)
    res = strftime('%Y-%m-%d %H:%M:%S', timestruct)
    # print('\n\n\n aaaaaaaaaaaaaaaaaa'+res+'\n\n\n')
    return Response({'code':200, 'message':'success','data':res})


@api_view(["POST"])
def text_recognize(request):

    projid = request.POST.get('project')
    position =json.loads(request.POST['position'])

    pic = Project.objects.get(id=projid).picture.picture
    path=re.sub('TemplateOCR','static',os.path.dirname(os.path.abspath(__file__)))+pic.url
    path=unquote(path)
    print('【path】',path)
    img = cv2.imread(path)
    print('【img】',img.shape)
    points = transPoint(position)
    x1 = points[0][0]
    x2 = points[1][0]
    y1 = points[0][1]
    y2 = points[3][1]
    img_sub = img[y1:y2, x1:x2, :]
    res=recognizer(text_recognizer,img_sub)[0][0]

    return Response({'code':200, 'message':'success','data':res})



@api_view(["POST"])
def upload_img(request):

    pic_file = request.FILES.get('img')
    if 'jpeg' not in pic_file.name and 'pdf' not in pic_file.name  and 'JPG' not in pic_file.name and 'JPEG' not in pic_file.name and 'PNG' not in pic_file.name  and 'png' not in pic_file.name and 'jpg' not in pic_file.name:
        return Response({'code':400, 'message':pic_file.name+'不是图像，请上传jpeg,png,pdf格式图像'}, status=400)
    if 'pdf' in pic_file.name:
        pdf_file = request.FILES.get('img')
        pdf_doc = Bulk_pdf(pdf_file=pdf_file)
        pdf_doc.save()
        pdfDoc = fitz.open(pdf_doc.pdf_file.path)
        for pg in range(len(pdfDoc)):
            page = pdfDoc[pg]
            rotate = int(0)
            zoom_x = 2
            zoom_y = 2
            mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            pix.save(
                 os.path.dirname( os.path.dirname(pdf_doc.pdf_file.path))+r'/bulk_img' + '/' + str(
                    pdf_file.name) + '_Page'+str(pg)+ '.jpg')
    else:
        picture = Bulk_picture()
        pic_io = BytesIO()
        img = Image.open(pic_file)
        pic = InMemoryUploadedFile(
            file=pic_file,
            field_name=None,
            name=pic_file.name,
            content_type=pic_file.content_type,
            size=img.size,
            charset=None
        )

        picture.picture = pic
        picture.picture_name = picture.picture.name
        try:
            picture.save()
        except OSError as e:
            print('cannot save image, ', e)
            return Response({'code': 200, 'message': 'success'})
            # return Response({'code': 500, 'message': 'failed'})

    return Response({'code':200, 'message':'success'})


@api_view(["POST"])
def bulk_predictor(request):
    model_type=request.POST.get('model_type')
    path1 = os.path.dirname(os.path.abspath(__file__))
    bulk_file_path = os.path.dirname(path1) + r'/static/media/template_file/bulk_img/'
    total_res1=[]
    total_ans=[]
    modeltype = Moudeltype.objects.get(model_type=model_type)
    batch=Exportresult_bulk.objects.all().aggregate(Max('batch'))['batch__max']
    if batch == None:
        batch = 0
    else:
        batch = batch + 1
    bulk_exportres_form = []
    for filename in os.listdir(bulk_file_path):
        img_path=os.path.join(bulk_file_path,filename)
        img=cv2.imread(img_path)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img_ = redink_remover(img)
        res, res_img, ans  = SER_generator(img_)#(img,text_sys,kie_predictor)
        total_res1.append(res)
        total_ans.append(ans)

        for subans in ans:
            temp2db={'question':subans['Q'], 'answer':subans['A'], 'create_time':timezone.now(),'last_mod_time':timezone.now(),
                     'moudeltype_name': modeltype.model_type_name,'pic_name':re.split(r'[\\/]',filename)[-1],'batch':batch,'model_type':modeltype.model_type}
            bulk_exportres_form.append(temp2db)


    bulk_form=[Exportresult_bulk(question=x['question'],answer=x['answer'],create_time=x['create_time'],
                                           last_mod_time=x['last_mod_time'],modeltype_name=x['moudeltype_name'],
                                           pic_name=x['pic_name'],batch=x['batch'],model_type=x['model_type'] ) for x in bulk_exportres_form]
    Exportresult_bulk.objects.bulk_create(bulk_form)

    allpdf=Bulk_pdf.objects.all()
    allpdf.delete()


    path1 = os.path.dirname(os.path.abspath(__file__))
    img_file_path = os.path.dirname(path1) + r'/static/media/template_file/bulk_img/'
    move2img_file_path = os.path.dirname(path1) + r'/static/media/template_file/bulk_img2/'
    imgs = os.listdir(img_file_path)
    try:
        for m in imgs:
            if m in os.listdir(move2img_file_path):
                os.remove(img_file_path + m)
            else:
                shutil.move(img_file_path + m,move2img_file_path)
    except OSError as e:
        print('cannot remove imgs', e)

    pdf_file_path = os.path.dirname(path1) + r'/static/media/template_file/bulk_pdf/'
    pdfs = os.listdir(pdf_file_path)
    try:
        for m in pdfs:
            os.remove(pdf_file_path + m)
    except OSError as e:
        print('cannot delete pdfs', e)
    # export_data=Exportresult_bulk.objects.all().values()
    # export_data = Exportresult_bulk.objects.filter(batch=batch).values()
    # ptr = Paginator(export_data, 100)
    # masters = ptr.page(1)
    # export_res = Exportresult_bulkSerializer(masters, many=True)
    return Response({'code': 200, 'message': 'success','data':batch})




@api_view(["GET", "POST"])
def get_bulk_res(request):
    # 获取分页参数
    page = request.GET.get('pageNum')
    pageSize = request.GET.get('pageSize')
    
    # 获取过滤参数
    moudeltype_name = request.GET.get('moudeltype_name')  # 模型类型
    datetimerange_start = request.GET.get('datetimerangeStart')  # 开始时间
    datetimerange_end = request.GET.get('datetimerangeEnd')  # 结束时间

    # 打印接收到的参数，方便调试
    print('=== 查询参数 ===')
    print(f'moudeltype_name: {moudeltype_name}')
    print(f'datetimerangeStart: {datetimerange_start}')
    print(f'datetimerangeEnd: {datetimerange_end}')
    print('================')

    # 设置默认分页参数
    if not page:
        page = 1
    if not pageSize or pageSize == 0:
        pageSize = 100000

    # 构建查询条件
    project_list = Exportresult_bulk.objects.all()
    
    # 按模型类型过滤
    if moudeltype_name and moudeltype_name.strip() and moudeltype_name != 'undefined':
        print(f'按模型类型过滤: {moudeltype_name}')
        project_list = project_list.filter(model_type=moudeltype_name)
    
    # 按时间范围过滤（使用 last_mod_time 字段）
    # 时间范围是一个区间：datetimerangeStart <= last_mod_time <= datetimerangeEnd
    if datetimerange_start and datetimerange_start.strip() and datetimerange_start != 'undefined':
        try:
            # 处理不同格式的时间字符串
            start_time_str = datetimerange_start.replace('Z', '+00:00')
            # 尝试解析 ISO 格式
            if 'T' in start_time_str:
                start_time = datetime.datetime.fromisoformat(start_time_str)
            else:
                # 如果是简单的日期格式
                start_time = datetime.datetime.strptime(start_time_str, '%Y-%m-%d')
            
            print(f'开始时间过滤: >= {start_time}')
            project_list = project_list.filter(last_mod_time__gte=start_time)
        except (ValueError, AttributeError) as e:
            print(f'解析开始时间失败: {e}')
    
    if datetimerange_end and datetimerange_end.strip() and datetimerange_end != 'undefined':
        try:
            # 处理不同格式的时间字符串
            end_time_str = datetimerange_end.replace('Z', '+00:00')
            # 尝试解析 ISO 格式
            if 'T' in end_time_str:
                end_time = datetime.datetime.fromisoformat(end_time_str)
            else:
                # 如果是简单的日期格式，设置为当天的23:59:59
                end_time = datetime.datetime.strptime(end_time_str, '%Y-%m-%d')
                end_time = end_time.replace(hour=23, minute=59, second=59)
            
            print(f'结束时间过滤: <= {end_time}')
            project_list = project_list.filter(last_mod_time__lte=end_time)
        except (ValueError, AttributeError) as e:
            print(f'解析结束时间失败: {e}')

    print(f'过滤后数据量: {project_list.count()}')

    res = {}
    res['code'] = 20000
    res['msg'] = 'success'
    res['data'] = {'alldata':project_list.values(),'ids':project_list.values_list('id',flat=True),'length': project_list.count(),'pageNum': int(page),'pageSize': int(pageSize)}
    return Response(res)


# ...existing code...
@api_view(["POST"])
def bulk_predictor_http(request):
    """
    支持三种 inputs:
      - 上传的文件（multipart/form-data，字段名 'files'，可多个）
      - 远程文件 URL 列表（字段名 'files'，支持 HTTP/HTTPS，可为 JSON 字符串或逗号分隔）
      - 混合模式：同时支持上传文件和 URL
    
    支持的 URL 格式示例：
      - http://10.2.13.68:8889/pdfsample.pdf
      - https://example.com/document.pdf
      - https://example.com/image.jpg
    
    对于每个输入文件：
      - 如果是 PDF：写临时文件并用 convert_pdf_with_adaptive_size 转为多张图，再对每页调用 SER_generator
      - 如果是图片：直接调用 SER_generator
    
    返回: {'code':200,'message':'success','data':[ {'filename':xxx,'results': [ [ {Q,A},... ], ... ], 'error':opt}, ... ]}
    """
    try:
        # 创建支持 HTTPS 的 SSL 上下文
        ssl_context = ssl.create_default_context()
        # 如果需要支持自签名证书（仅用于内网测试环境），可以取消下面两行注释
        # ssl_context.check_hostname = False
        # ssl_context.verify_mode = ssl.CERT_NONE
        
        # 收集输入项：优先从 request.FILES，再从 request.data 中解析
        items = []
        try:
            # request.FILES.getlist 支持多文件上传
            items = request.FILES.getlist('files') or []
        except Exception:
            try:
                items = request.data.getlist('files') or []
            except Exception:
                items = []

        # 如果没有通过 multipart 上传的文件，尝试解析请求体中的 'files' 字段（JSON 列表或逗号分隔字符串）
        if not items:
            raw = request.data.get('files') or request.POST.get('files')
            if raw:
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        items = parsed
                    else:
                        items = [parsed]
                except Exception:
                    # 逗号分隔形式或单个 URL 字符串
                    items = [u.strip() for u in str(raw).split(',') if u.strip()]

        if not items:
            return Response({'code': 400, 'message': "no files provided (expect 'files' list)"}, status=400)

        results = []
        import tempfile
        for it in items:
            entry = {'filename': None, 'results': []}
            try:
                # 判断是字符串（URL）还是文件对象
                if isinstance(it, str):
                    url = it.strip()
                    # 验证 URL 格式
                    if not (url.startswith('http://') or url.startswith('https://')):
                        entry['error'] = f"Invalid URL format: {url}. Must start with http:// or https://"
                        results.append(entry)
                        continue
                    
                    # 从 URL 中提取文件名
                    filename = os.path.basename(url.split('?')[0])
                    if not filename:
                        filename = 'remote_file'
                    entry['filename'] = filename
                    
                    # 根据协议选择是否使用 SSL 上下文
                    try:
                        if url.startswith('https://'):
                            print(f'[HTTPS] 正在下载文件: {url}')
                            with urllib.request.urlopen(url, timeout=60, context=ssl_context) as resp:
                                content = resp.read()
                        else:
                            print(f'[HTTP] 正在下载文件: {url}')
                            with urllib.request.urlopen(url, timeout=60) as resp:
                                content = resp.read()
                        print(f'[下载完成] 文件大小: {len(content)} bytes')
                    except urllib.error.HTTPError as e:
                        entry['error'] = f"HTTP Error {e.code}: {e.reason} - {url}"
                        results.append(entry)
                        continue
                    except urllib.error.URLError as e:
                        entry['error'] = f"URL Error: {str(e.reason)} - {url}"
                        results.append(entry)
                        continue
                    except Exception as e:
                        entry['error'] = f"Download failed: {str(e)} - {url}"
                        results.append(entry)
                        continue
                        
                elif hasattr(it, 'read'):
                    # Django 上传的文件或任意 file-like 对象
                    file_obj = it
                    filename = getattr(file_obj, 'name', 'unnamed')
                    entry['filename'] = filename
                    print(f'[上传文件] 文件名: {filename}')
                    content = file_obj.read()
                else:
                    # 未知类型，尝试转为字符串 URL
                    url = str(it).strip()
                    if not (url.startswith('http://') or url.startswith('https://')):
                        entry['error'] = f"Invalid input type or URL format: {url}"
                        results.append(entry)
                        continue
                    
                    filename = os.path.basename(url.split('?')[0])
                    if not filename:
                        filename = 'remote_file'
                    entry['filename'] = filename
                    
                    try:
                        if url.startswith('https://'):
                            with urllib.request.urlopen(url, timeout=60, context=ssl_context) as resp:
                                content = resp.read()
                        else:
                            with urllib.request.urlopen(url, timeout=60) as resp:
                                content = resp.read()
                    except Exception as e:
                        entry['error'] = f"Download failed: {str(e)} - {url}"
                        results.append(entry)
                        continue

                # 处理 PDF
                if entry['filename'] and entry['filename'].lower().endswith('.pdf'):
                    print(f'[处理 PDF] {entry["filename"]}')
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tf:
                        tf.write(content)
                        tmp_pdf = tf.name
                    try:
                        pages = convert_pdf_with_adaptive_size(tmp_pdf)
                        print(f'[PDF 转换] 共 {len(pages)} 页')
                        for idx, pg in enumerate(pages):
                            print(f'[处理页面] {idx + 1}/{len(pages)}')
                            if isinstance(pg, Image.Image):
                                img_cv = cv2.cvtColor(np.asarray(pg), cv2.COLOR_RGB2BGR)
                            else:
                                img_cv = pg
                            img_proc = redink_remover(img_cv)
                            _, _, ans = SER_generator(img_proc)
                            entry['results'].append(ans)
                    finally:
                        try:
                            os.remove(tmp_pdf)
                        except Exception:
                            pass
                else:
                    # 处理图片（Bytes -> PIL -> cv2）
                    print(f'[处理图片] {entry["filename"]}')
                    img = Image.open(BytesIO(content))
                    img = rotate(img)
                    img_cv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                    img_proc = redink_remover(img_cv)
                    _, _, ans = SER_generator(img_proc)
                    entry['results'].append(ans)
                
                print(f'[处理完成] {entry["filename"]} - 结果数: {len(entry["results"])}')

            except Exception as e:
                import traceback
                entry['error'] = f"{str(e)}\n{traceback.format_exc()}"
                print(f'[错误] 处理文件失败: {entry.get("filename", "unknown")} - {str(e)}')
            
            results.append(entry)

        return Response({'code': 200, 'message': 'success', 'data': results})
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f'[严重错误] {error_msg}')
        return Response({'code': 500, 'message': error_msg}, status=500)
# ...existing code...


@api_view(["POST"])
def export_bulk(request):
    col_names = ['id', 'question', 'answer', 'create_time', 'modeltype_name','pic_name']
    project_list = Exportresult_bulk.objects.all().values(*col_names)
    return Response({'code': 200, 'message': 'success','data':project_list})





@api_view(["POST"])
def delete_all_bulk(request):
    ids=json.loads(request.POST['ids'])
    exportresult_bulk=Exportresult_bulk.objects.filter(id__in=ids)
    exportresult_bulk.delete()

    return Response({'code': 200, 'message': 'success'})


@api_view(["PUT"])
def update_bulk(request):
    new_data = request.query_params.copy()
    bulk_form = Exportresult_bulkFrom(new_data)
    if bulk_form.is_valid():
        Exportresult_bulk.objects.filter(id=new_data['id']).update(
            question=new_data['question'],
            answer=new_data['answer'],
            last_mod_time=timezone.now(),
            modeltype_name=new_data['modeltype_name']   # 修正拼写
        )
        return Response({'code': 200, 'message': '修改成功！'})
    return Response({'code': 403, 'message': '请确认数据输入正确（此数据已经存在）'}, status=403)



@api_view(['delete'])
def bulk_delete(request):
    res = Exportresult_bulk.objects.get(id=request.query_params['id'])
    # mark = Mark.objects.get(id=request.query_params['id'])
    res.delete()
    return Response({'code': 200, 'message': '删除成功'})


@api_view(["POST"])
def delete_all_trainset(request):
    markresults = Markresults.objects.all()
    markresults.delete()
    return Response({'code': 200, 'message': 'success'})






@api_view(["POST"])
def contractData(request):
    data = request.FILES.values()

    pred = pd.DataFrame()
    truth = pd.DataFrame()
    
    # 定义期望的pred文件列名（只检查pred文件格式）
    expected_pred_columns = ['question', 'answer', 'pic_name']
    expected_pred_columns_new = ['项目', '内容', '图片名称']  # 新格式
    
    for f in data:
        print('\n' + '='*80)
        print(f'【开始处理文件】')
        print(f'  - 文件名: {f.name}')
        print(f'  - 字段名: {f.field_name}')
        print(f'  - 文件大小: {f.size} bytes')
        print('='*80)
        
        try:
            temp = pd.read_excel(f)
            print(f'✓ Excel读取成功')
        except Exception as e:
            print(f'✗ 【读取Excel失败】: {e}')
            return Response({
                'code': 400,
                'message': f'读取Excel文件失败: {str(e)}',
                'data': []
            })
        
        print(f'\n【步骤1：原始列名】')
        print(f'  列数: {len(temp.columns)}')
        for idx, col in enumerate(temp.columns):
            print(f'  列{idx}: "{col}" | 类型:{type(col).__name__} | 长度:{len(str(col))} | repr:{repr(col)}')
        print(f'  数据行数: {len(temp)}')
        
        # 清理列名：去除前后空格和换行符
        temp.columns = temp.columns.str.strip().str.replace('\n', '').str.replace('\r', '')
        
        print(f'\n【步骤2：清理后列名】')
        for idx, col in enumerate(temp.columns):
            print(f'  列{idx}: "{col}" | 长度:{len(str(col))}')
        
        if f.field_name.startswith('predfile'):
            print(f'\n【步骤3：predfile文件 - 开始列名匹配】')
            
            # 宽松匹配列名（忽略大小写、空格、特殊字符）
            cols_clean = {}
            for col in temp.columns:
                clean_key = str(col).strip().lower().replace(' ', '').replace('\n', '').replace('\r', '')
                cols_clean[clean_key] = col
                print(f'  原始列名 "{col}" -> 清理后键 "{clean_key}"')
            
            # 尝试多种可能的列名变体
            question_col = None
            answer_col = None
            pic_name_col = None
            
            print(f'\n【步骤4：匹配"项目"列】')
            for key in cols_clean:
                print(f'  检查键: "{key}"', end='')
                if key in ['项目', 'xiangmu', 'question', 'xm']:
                    question_col = cols_clean[key]
                    print(f' -> ✓ 匹配成功! 原始列名: "{question_col}"')
                    break
                else:
                    print(f' -> ✗ 不匹配')
            
            print(f'\n【步骤5：匹配"内容"列】')
            for key in cols_clean:
                print(f'  检查键: "{key}"', end='')
                if key in ['内容', 'neirong', 'answer', 'nr']:
                    answer_col = cols_clean[key]
                    print(f' -> ✓ 匹配成功! 原始列名: "{answer_col}"')
                    break
                else:
                    print(f' -> ✗ 不匹配')
            
            print(f'\n【步骤6：匹配"图片名称"列】')
            for key in cols_clean:
                print(f'  检查键: "{key}"', end='')
                if key in ['图片名称', 'tupianmingcheng', 'pic_name', 'picname', 'tpmc', '图片']:
                    pic_name_col = cols_clean[key]
                    print(f' -> ✓ 匹配成功! 原始列名: "{pic_name_col}"')
                    break
                else:
                    print(f' -> ✗ 不匹配')
            
            print(f'\n【步骤7：最终匹配结果】')
            print(f'  项目列: {question_col if question_col else "❌ 未匹配"}')
            print(f'  内容列: {answer_col if answer_col else "❌ 未匹配"}')
            print(f'  图片名称列: {pic_name_col if pic_name_col else "❌ 未匹配"}')
            
            if question_col and answer_col and pic_name_col:
                print('【成功匹配列名，开始转换】')
                temp = temp.rename(columns={
                    question_col: 'question',
                    answer_col: 'answer',
                    pic_name_col: 'pic_name'
                })
                temp = temp[['question', 'answer', 'pic_name']]
                print(f'【转换后列名】: {temp.columns.tolist()}')
            else:
                # 提供更详细的错误信息
                print('【格式匹配失败，详细信息】:')
                for i, col in enumerate(temp.columns):
                    print(f'  列 {i}: "{col}" (类型: {type(col)}, 长度: {len(str(col))}, repr: {repr(col)})')
                print(f'【清理后的列名键】: {list(cols_clean.keys())}')
                return Response({
                    'code': 400,
                    'message': f'pred文件格式错误！期望包含列: {expected_pred_columns_new} 或 {expected_pred_columns}，实际列: {temp.columns.tolist()}。未能匹配到: {"项目" if not question_col else ""}{"内容" if not answer_col else ""}{"图片名称" if not pic_name_col else ""}',
                    'data': []
                })
            pred = pd.concat([pred, temp], ignore_index=True)
            
        else:  # truthfile
            # truth文件不做格式检查，直接读取
            print('【truth文件不做格式检查，直接使用】')
            truth = pd.concat([truth, temp], ignore_index=True)
    
    print(f'【pred数据形状】: {pred.shape}, 列名: {pred.columns.tolist()}')
    print(f'【truth数据形状】: {truth.shape}, 列名: {truth.columns.tolist()}')
    
    # 检查数据是否为空
    if pred.empty:
        return Response({'code': 400, 'message': 'pred文件没有数据！', 'data': []})
    if truth.empty:
        return Response({'code': 400, 'message': 'truth文件没有数据！', 'data': []})

    items = ['企业名称', '法定代表人', '注册地址', '经营范围', '统一社会信用代码',
             '成立日期', '注册资本', '企业类型', '营业期限', 'pic_name']

    newdata = {k: [] for k in items}

    def safe_get(item_df, question):
        """安全获取 item_df 中 question 的 answer，如果不存在则返回空"""
        res = item_df[item_df['question'] == question]
        if not res.empty:
            return res['answer'].iloc[0]
        return ''

    for name, item in pred.groupby('pic_name'):
        questions = item['question'].tolist()
        for field in items:
            if field == 'pic_name':
                newdata[field].append(name)
                continue

            if field in questions:
                newdata[field].append(safe_get(item, field))
            elif field == '企业名称':
                newdata[field].append(safe_get(item, '名称'))
            elif field == '成立日期':
                newdata[field].append(safe_get(item, '注册日期'))
            elif field == '注册地址':
                newdata[field].append(safe_get(item, '经营场所') or safe_get(item, '住所'))
            elif field == '企业类型':
                newdata[field].append(safe_get(item, '类型'))
            elif field == '法定代表人':
                val = safe_get(item, '负责人') or safe_get(item, '经营者')
                newdata[field].append(val)
            else:
                newdata[field].append('')

    newPreddata = pd.DataFrame(newdata)

    newPreddata['name'] = newPreddata['企业名称'].str.replace(r'[^\w\d\n ]', '', regex=True)
    newTruthdata = truth[items[:-1]].copy()
    newTruthdata['name'] = newTruthdata['企业名称'].str.replace(r'[^\w\d\n ]', '', regex=True)

    intersection = pd.merge(newPreddata, newTruthdata, on='name', how='inner').fillna('_')

    # 检查是否有匹配的数据
    if len(intersection) == 0:
        print(f'【警告】pred和truth没有匹配的企业！')
        print(f'  pred企业名称: {newPreddata["企业名称"].tolist()[:5]}...')  # 显示前5个
        print(f'  truth企业名称: {newTruthdata["企业名称"].tolist()[:5]}...')  # 显示前5个
        return Response({
            'code': 200, 
            'message': '预测数据和真实数据没有匹配的企业。请检查企业名称是否一致。', 
            'data': [],
            'others': {
                'truthdataNum': len(newTruthdata),
                'preddataNum': len(newPreddata),
                'alignedNum': 0,
                'codeCorrectness': 0,
                'total_acc': 0
            }
        })

    def normalize_date(val):
        if val=='长期': return '长期'
        if pd.isna(val) or val=='-': return '00'
        val = re.sub(r'[年月]', '-', str(val))
        val = re.sub(r'[日 \n]', '', val)
        parts = val.split('-')
        return '-'.join([p.zfill(2) for p in parts])
    def normalize_others(val):
        if val=='-': return '-'
        if pd.isna(val) : return '-'
        
        return val

    for col in ['成立日期', '营业期限']:
        newTruthdata[col] = newTruthdata[col].apply(normalize_date)
        newPreddata[col] = newPreddata[col].apply(normalize_date)

    def clean_text(val):
        return re.sub(r'[ \n]', '', str(val)).replace('）', ')').replace('（', '(')

    for col in items[:-1]:
        if col != '企业名称':
            newTruthdata[col] = newTruthdata[col].map(clean_text)
            newPreddata[col] = newPreddata[col].map(clean_text)
    newTruthdata['注册资本']=newTruthdata['注册资本'].map(normalize_others)
    newPreddata['注册资本']=newPreddata['注册资本'].map(normalize_others)
    # 比对分析
    tableres = []
    total_acc = 0
    code_correct = 0

    for _, row in intersection.iterrows():
        truth_row = newTruthdata[newTruthdata['企业名称'] == row['企业名称_y']].iloc[-1].fillna('-')
        pred_row = newPreddata[newPreddata['企业名称'] == row['企业名称_x']].iloc[-1].fillna('-')

        truth_row = truth_row.replace('nan', '-')
        truth_row = truth_row.replace('', '-')
        pred_row = pred_row.replace('nan', '-')
        pred_row = pred_row.replace('', '-')

        temp = {
            '企业名称_y': row['企业名称_y'],
            '统一社会信用代码_y': row['统一社会信用代码_y'],
            'family': []
        }

        t_dict = truth_row.to_dict()
        p_dict = pred_row.to_dict()

        acc_count = 0
        for key in t_dict:
            if key not in p_dict or key in['name','企业名称']:
                continue

            t_clean = re.sub(r'[^\w\d ]', '', str(t_dict[key]))
            p_clean = re.sub(r'[^\w\d ]', '', str(p_dict[key]))
            if t_clean!=p_clean:
                print(key,'【'+row['企业名称_y']+'】',t_clean,p_clean,len(t_dict),t_dict)
            tag = '1' if t_clean == p_clean else '0'
            p_dict[key + '_tag'] = tag
            # if tag == '1':
            #     acc_count += 1
            score = char_overlap_rate(p_clean, t_clean)
            p_dict[key + '_score'] = round(score, 3) 
            acc_count += score

        if t_dict['统一社会信用代码'] == p_dict['统一社会信用代码']:
            code_correct += 1

        t_dict['数据来源'] = '真实'
        p_dict['数据来源'] = '预测'
        t_dict['经营范围0'] = t_dict['经营范围'][:15]
        p_dict['经营范围0'] = p_dict['经营范围'][:15]

        temp['family'].append(t_dict)
        temp['family'].append(p_dict)
        # temp['acc'] = round(acc_count / len(t_dict), 2)
        temp['acc'] = round(acc_count / 8, 3)
        total_acc += temp['acc']
        tableres.append(temp)

    otherRes = {
        'truthdataNum': len(newTruthdata),
        'preddataNum': len(newPreddata),
        'alignedNum': len(intersection),
        'codeCorrectness': round(code_correct / len(intersection), 2) if len(intersection) > 0 else 0,
        'total_acc': round(total_acc / len(intersection), 2) if len(intersection) > 0 else 0
    }

    return Response({'code': 200, 'message': 'success', 'data': tableres, 'others': otherRes})




 
@api_view(['GET'])
def resetmodel(request):
    try:
        ori_model_dir_det=os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))) + '/data/inital_models/det2'
        target_model_dir_det= os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))) +'/TemplateOCR/Funcs/inferenced_models/det2'
        shutil.copytree(ori_model_dir_det,target_model_dir_det,dirs_exist_ok=True)

        ori_model_dir_ser=os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))) + '/data/inital_models/ser'
        target_model_dir_ser= os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))) +'/TemplateOCR/Funcs/inferenced_models/ser'
        shutil.copytree(ori_model_dir_ser,target_model_dir_ser,dirs_exist_ok=True)
        yml_ser_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r'/static/media/template_file/backup-ser.yml'
        yml_ser_path_dst=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r'/static/media/template_file/ser.yml'
        yml_det_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r'/static/media/template_file/backup-det.yml'
        yml_det_path_dst=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r'/static/media/template_file/det.yml'
        shutil.copyfile(yml_ser_path,yml_ser_path_dst)
        shutil.copyfile(yml_det_path,yml_det_path_dst)
        mod1=importlib.import_module("Funcs.bulkrecfuncs.modelfunc")
        mod2=importlib.import_module("task")
        importlib.reload(mod1)
        importlib.reload(mod2)
        from Funcs.bulkrecfuncs.modelfunc import SER_generator,kie_predictor ,text_sys ,convert_pdf_with_adaptive_size, clear_directory
        from task import async_task, REC_Model,DET_Model, recognizer,generat_class_file

        text_recognizer=REC_Model()
        text_detector=DET_Model()
        reload_model()
        # SerPredictor=kie_predictor #SerPredictor_Model()


    except Exception as e:
        print(e)
        return Response({'code': 500, 'message': 'failed'})
    return Response({'code': 200, 'message': 'success'})


@api_view(['POST'])
def refresh_model_status(request):
    """Endpoint to reset a training task's status and optionally record an error.

    Expected POST body (form or JSON):
      - task_id: integer (default 0)
      - error: optional string error message

    This updates the `TemplateOCR_traintask` table similarly to the other
    update_status implementations used in the project.
    """
    try:
        data = request.data if hasattr(request, 'data') else request.POST
        task_id = int(data.get('task_id', 0))
        error = data.get('error', '')

        DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'db.sqlite3')
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        updates = []
        params = []
        if error:
            updates.append("status = ?")
            params.append('error')
            updates.append("progress = ?")
            params.append(0)
            updates.append("error_msg = ?")
            params.append(error)
        else:
            updates.append("status = ?")
            params.append('idle')
            updates.append("progress = ?")
            params.append(0)
            updates.append("error_msg = ?")
            params.append('')

        updates.append("last_update = datetime('now')")
        sql = f"UPDATE TemplateOCR_traintask SET {', '.join(updates)} WHERE id = ?"
        params.append(task_id)

        cursor.execute(sql, params)
        conn.commit()
        conn.close()

        return Response({'code': 200, 'message': 'ok'})
    except Exception as e:
        return Response({'code': 500, 'message': str(e)}, status=500)

