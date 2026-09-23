import os,sys

path1=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append( path1+r'\TemplateOCR\Funcs\PaddleOCRrelease\PaddleOCRrelease')
sys.path.append( '/data/tensorflow/miniforge3/envs/paddle_env/lib/python3.8/site-packages/paddleocr')
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
# from tools.infer.predict_system import TextSystem as TextSystem
# from tools.infer_kie_token_ser import SerPredictor as SerPredictor
# import tools.export_model.main as export_main
import cmapy
import tools.export_model as export_model

from tools.train import main as train_main
from ppocr.utils.utility import get_image_file_list, check_and_read
import cv2
import time
import argparse
from ppocr.utils.utility import set_seed
import shutil
import re
import json
import random
import copy
import numpy as np
import importlib.util
import sqlite3
import yaml
import random
import paddle
import math



print('[function] path:', os.path.join(os.path.dirname(os.path.abspath(__file__))))
DET_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Funcs/inferenced_models/det2')
REC_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Funcs/inferenced_models/ch_PP-OCRv4_rec_infer')
CLS_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Funcs/inferenced_models/ch_ppocr_mobile_v2.0_cls_infer')
KIE_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Funcs/inferenced_models/ser/inference')
SER_CLASS_DIR=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'static','media','template_file','allLabel_tax.txt')
CLASS_FILE=SER_CLASS_DIR



# DET_MODEL_DIR = '/data/tensorflow/kath/Service/LicenseRecInterface/models/det2'
# REC_MODEL_DIR = '/data/tensorflow/kath/Service/LicenseRecInterface/models/ch_PP-OCRv4_rec_infer'
# CLS_MODEL_DIR='/data/tensorflow/kath/Service/LicenseRecInterface/models/ch_ppocr_mobile_v2.0_cls_infer'


DB_PATH= os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'db.sqlite3')

def update_status(task_id, status=None, progress=None, error=''):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    updates = []
    params = []

    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if progress is not None:
        updates.append("progress = ?")
        params.append(progress)
    if error is not None:
        updates.append("error_msg = ?")
        params.append(error)

    updates.append("last_update = datetime('now')")
    sql = f"UPDATE TemplateOCR_traintask SET {', '.join(updates)} WHERE id = ?"
    params.append(task_id)

    cursor.execute(sql, params)
    conn.commit()
    conn.close()




def load_module_from(path, name):
    file_path = os.path.join(path, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module) 
    return module



def get_image_file_list(resfile_path):
    lst=[]
    with open(resfile_path, 'r') as f:
        for line in f:
            lst.append(line.split('\t')[0])
    return lst



my_arg = argparse.Namespace(use_gpu=True, use_xpu=False, use_npu=False, ir_optim=True, use_tensorrt=False, min_subgraph_size=15, precision='fp32', 
  gpu_mem=500, gpu_id=0, image_dir='', page_num=0, det_algorithm='CT', det_model_dir=path1+'/TemplateOCR/Funcs/inferenced_models/det2', 
  det_ct_score_thresh=0.3,
  det_limit_side_len=1280, det_limit_type='max', det_box_type='quad', det_db_thresh=0.2, det_db_box_thresh=0.3, 
  det_db_unclip_ratio=1.5, max_batch_size=10, use_dilation=False, det_db_score_mode='fast', det_east_score_thresh=0.8, det_east_cover_thresh=0.1, 
  det_east_nms_thresh=0.2, det_sast_score_thresh=0.5, det_sast_nms_thresh=0.2, det_pse_thresh=0, det_pse_box_thresh=0.85, det_pse_min_area=16, 
  det_pse_scale=1, scales=[8, 16, 32], alpha=1.0, beta=1.0, fourier_degree=5, rec_algorithm='SVTR_LCNet', rec_model_dir=path1+'/TemplateOCR/Funcs/inferenced_models/ch_PP-OCRv4_rec_infer', rec_image_inverse=True, rec_image_shape='3, 48, 320', rec_batch_num=6, max_text_length=25, rec_char_dict_path='/data/tensorflow/kath/OCR/PaddleOCR-release-2.7/ppocr/utils/ppocr_keys_v1.txt', use_space_char=True, vis_font_path='./doc/fonts/simfang.ttf', drop_score=0.5, e2e_algorithm='PGNet', e2e_model_dir=None, e2e_limit_side_len=768, e2e_limit_type='max', e2e_pgnet_score_thresh=0.5, e2e_char_dict_path='./ppocr/utils/ic15_dict.txt', e2e_pgnet_valid_set='totaltext', e2e_pgnet_mode='fast', use_angle_cls=False, cls_model_dir='TemplateOCR/Funcs/inferenced_models/cls', cls_image_shape='3, 48, 192', label_list=['0', '180'], cls_batch_num=6, cls_thresh=0.9, enable_mkldnn=False, cpu_threads=10, use_pdserving=False, warmup=False, sr_model_dir=None, sr_image_shape='3, 32, 128', sr_batch_num=1, draw_img_save_dir='./inference_results', save_crop_res=False, crop_res_save_dir='./output', use_mp=False, total_process_num=1, process_id=0, benchmark=False, save_log_path='./log_output/', show_log=True, use_onnx=False)

SERCONF={'Global':
                 {'use_gpu': True, 'epoch_num': 200, 'log_smooth_window': 10, 'print_batch_step': 10, 
                  'save_model_dir': './output/BigModel1', 'save_epoch_step': 2000, 'eval_batch_step': [0, 59],
                  'cal_metric_during_train': False, 'save_inference_dir': None, 'use_visualdl': False,
                  'seed': 2022, 'infer_img': '',
                  'save_res_path': '/data/tensorflow/kath/Service/LegalDepartment/OCRProject/backend/TemplateOCR/Funcs/bulkrecfuncs/', 'kie_rec_model_dir':REC_MODEL_DIR,
                  'kie_det_model_dir': DET_MODEL_DIR, 'infer_mode': False,
                  'det_algorithm': 'CT', 'det_model_dir': DET_MODEL_DIR,
                  'rec_model_dir': REC_MODEL_DIR,
                  'use_angle_cls':False,
                  'cls_model_dir': CLS_MODEL_DIR, 'kie_cls_model_dir': CLS_MODEL_DIR,
                  'class_path':'/data/tensorflow/kath/Service/LegalDepartment/OCRProject/backend/static/media/template_file/allLabel_tax.txt',
                  'distributed': False},
             'Architecture': {'model_type': 'kie', 'algorithm': 'LayoutXLM', 'Transform': None,
                              'Backbone': {'name': 'LayoutXLMForSer', 'pretrained': '/data/tensorflow/kath/Service/LicenseRecInterface/models/ser/inference',
                                           'checkpoints': '/data/tensorflow/kath/Service/LicenseRecInterface/models/ser/inference',
                                           'mode': 'vi', 'num_classes': 11003}},
             'Loss': {'name': 'VQASerTokenLayoutLMLoss', 'num_classes': 11003, 'key': 'backbone_out'},
             'Optimizer': {'name': 'AdamW', 'beta1': 0.9, 'beta2': 0.999,
                           'lr': {'name': 'Linear', 'learning_rate': 5e-05, 'epochs': 200, 'warmup_epoch': 1},
                           'regularizer': {'name': 'L2', 'factor': 0.0}},
             'PostProcess': {'name': 'VQASerTokenLayoutLMPostProcess', 'class_path': '/data/tensorflow/kath/Service/LegalDepartment/OCRProject/backend/static/media/template_file/allLabel_tax.txt'},
             'Metric': {'name': 'VQASerTokenMetric', 'main_indicator': 'hmean'},
             'Train': {'dataset': {'name': 'SimpleDataSet', 'data_dir': '', 'label_file_list': [''], 'ratio_list': [1.0],
                                   'transforms': [{'DecodeImage': {'img_mode': 'RGB', 'channel_first': False}}, {'VQATokenLabelEncode': {'contains_re': False, 'algorithm': 'LayoutXLM', 'class_path': '/data/tensorflow/kath/Service/LegalDepartment/OCRProject/backend/static/media/template_file/allLabel_tax.txt', 'use_textline_bbox_info': True, 'order_method': 'tb-yx'}}, {'VQATokenPad': {'max_seq_len': 512, 'return_attention_mask': True}}, {'VQASerTokenChunk': {'max_seq_len': 512}}, {'Resize': {'size': [224, 224]}}, {'NormalizeImage': {'scale': 1, 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'order': 'hwc'}}, {'ToCHWImage': None}, {'KeepKeys': {'keep_keys': ['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'image', 'labels']}}]},
                       'loader': {'shuffle': True, 'drop_last': False, 'batch_size_per_card': 16, 'num_workers': 0}}, 'Eval': {'dataset': {'name': 'SimpleDataSet', 'data_dir': '/data/kath/data/Datasets/', 'label_file_list': ['/data/kath/data/Datasets/val.txt'], 'transforms': [{'DecodeImage': {'img_mode': 'RGB', 'channel_first': False}}, {'VQATokenLabelEncode': {'contains_re': False, 'algorithm': 'LayoutXLM', 'class_path':'/data/tensorflow/kath/Service/LegalDepartment/OCRProject/backend/static/media/template_file/allLabel_tax.txt', 'use_textline_bbox_info': True, 'order_method': 'tb-yx'}}, {'VQATokenPad': {'max_seq_len': 512, 'return_attention_mask': True}}, {'VQASerTokenChunk': {'max_seq_len': 512}}, {'Resize': {'size': [224, 224]}},
                                    {'NormalizeImage': {'scale': 0.8, 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'order': 'hwc'}}, {'ToCHWImage': None}, {'KeepKeys': {'keep_keys': ['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'image', 'labels']}}]},
                                                                                                                               'loader': {'shuffle': False, 'drop_last': False, 'batch_size_per_card': 16, 'num_workers': 0}}, 'profiler_options': None}


# SERCONF={'Global':
#                  {'use_gpu': False, 'epoch_num': 200, 'log_smooth_window': 10, 'print_batch_step': 10,
#                   'save_model_dir': './output/BigModel1', 'save_epoch_step': 2000, 'eval_batch_step': [0, 59],
#                   'cal_metric_during_train': False, 'save_inference_dir': None, 'use_visualdl': False,
#                   'seed': 2022, 'infer_img': '',
#                   'save_res_path': path1+"\\TemplateOCR\\Funcs\\outputs\\serRes", 'kie_rec_model_dir':path1+'/TemplateOCR/Funcs/inferenced_models/ch_PP-OCRv4_rec_infer',
#                   'kie_det_model_dir': path1+'\\TemplateOCR\\Funcs\\inferenced_models\\det2', 'infer_mode': False,
#                   'det_algorithm': 'CT', 'det_model_dir': path1+'\\TemplateOCR\\Funcs\\inferenced_models\\det2',
#                   'rec_model_dir': path1+'/TemplateOCR/Funcs/inferenced_models/ch_PP-OCRv4_rec_infer',
#                   'cls_model_dir': path1+'\\TemplateOCR\\Funcs\\inferenced_models\\cls', 'kie_cls_model_dir': path1+'\\TemplateOCR\\Funcs\\inferenced_models\\cls',
#                   'class_path': path1+'\\static\\media\\template_file\\allLabel_tax.txt',
#                   'distributed': False},
#              'Architecture': {'model_type': 'kie', 'algorithm': 'LayoutXLM', 'Transform': None,
#                               'Backbone': {'name': 'LayoutXLMForSer', 'pretrained': path1+'\\TemplateOCR\\Funcs\\inferenced_models\\ser\\inference',
#                                            'checkpoints': path1+'\\TemplateOCR\\Funcs\\inferenced_models\\ser\\inference',
#                                            'mode': 'vi', 'num_classes': 11003}},
#              'Loss': {'name': 'VQASerTokenLayoutLMLoss', 'num_classes': 11003, 'key': 'backbone_out'},
#              'Optimizer': {'name': 'AdamW', 'beta1': 0.9, 'beta2': 0.999,
#                            'lr': {'name': 'Linear', 'learning_rate': 5e-05, 'epochs': 200, 'warmup_epoch': 1},
#                            'regularizer': {'name': 'L2', 'factor': 0.0}},
#              'PostProcess': {'name': 'VQASerTokenLayoutLMPostProcess', 'class_path': path1+'\\static\\media\\template_file\\allLabel_tax.txt'},
#              'Metric': {'name': 'VQASerTokenMetric', 'main_indicator': 'hmean'},
#              'Train': {'dataset': {'name': 'SimpleDataSet', 'data_dir': '', 'label_file_list': [''], 'ratio_list': [1.0],
#                                    'transforms': [{'DecodeImage': {'img_mode': 'RGB', 'channel_first': False}}, {'VQATokenLabelEncode': {'contains_re': False, 'algorithm': 'LayoutXLM', 'class_path': path1+'\\static\\media\\template_file\\allLabel_tax.txt', 'use_textline_bbox_info': True, 'order_method': 'tb-yx'}}, {'VQATokenPad': {'max_seq_len': 512, 'return_attention_mask': True}}, {'VQASerTokenChunk': {'max_seq_len': 512}}, {'Resize': {'size': [224, 224]}}, {'NormalizeImage': {'scale': 1, 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'order': 'hwc'}}, {'ToCHWImage': None}, {'KeepKeys': {'keep_keys': ['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'image', 'labels']}}]},
#                        'loader': {'shuffle': True, 'drop_last': False, 'batch_size_per_card': 16, 'num_workers': 0}}, 'Eval': {'dataset': {'name': 'SimpleDataSet', 'data_dir': '/data/kath/data/Datasets/', 'label_file_list': ['/data/kath/data/Datasets/val.txt'], 'transforms': [{'DecodeImage': {'img_mode': 'RGB', 'channel_first': False}}, {'VQATokenLabelEncode': {'contains_re': False, 'algorithm': 'LayoutXLM', 'class_path': path1+'\\inferenced_models\\ser\\SWallLabel.txt', 'use_textline_bbox_info': True, 'order_method': 'tb-yx'}}, {'VQATokenPad': {'max_seq_len': 512, 'return_attention_mask': True}}, {'VQASerTokenChunk': {'max_seq_len': 512}}, {'Resize': {'size': [224, 224]}},
#                                     {'NormalizeImage': {'scale': 0.8, 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'order': 'hwc'}}, {'ToCHWImage': None}, {'KeepKeys': {'keep_keys': ['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'image', 'labels']}}]},
#                                                                                                                                'loader': {'shuffle': False, 'drop_last': False, 'batch_size_per_card': 16, 'num_workers': 0}}, 'profiler_options': None}



def REC_Model():
    t1=time.time()
    print('loading REC model...')
    text_recognizer = predict_rec.TextRecognizer(my_arg)
    t2 = time.time()
    print('Loading time for REC model: ',t2-t1)
    return text_recognizer


def DET_Model():
    t1=time.time()
    print('loading DET model...')
    text_recognizer = predict_det.TextDetector(my_arg)
    t2 = time.time()
    print('Loading time for DET model: ',t2-t1)
    return text_recognizer


# def SerPredictor_Model():
#     t1=time.time()
#     print('loading SerPredictor model...')
#     text_recognizer = SerPredictor(SERCONF)
#     t2 = time.time()
#     print('Loading time for SER model: ',t2-t1)
#     return text_recognizer


# def TextSystem_Model():
#     t1=time.time()
#     print('loading TextSystem model...')
#     text_recognizer = TextSystem(my_arg)
#     t2 = time.time()
#     print('Loading time for DET model: ',t2-t1)
#     return text_recognizer



def recognizer(REC_Model,img):
    rec_res, _ = REC_Model([img])
    return rec_res


def DET_retrainer(pretrained_dir,dataPath):
    import paddle
    # Dataset_generator
    print('[DET] Loading det model...')
    checkpoint=paddle.load(pretrained_dir+'.states')
    checkpoint['best_model_dict']['f_score']=0.75
    checkpoint['best_model_dict']['recall']=0.75
    checkpoint['best_model_dict']['precision']=0.75
    checkpoint['recall']=0.75
    checkpoint['precision']=0.75
    checkpoint['f_score']=0.75

    paddle.save(checkpoint,pretrained_dir+'.states')
    
    # 检测CUDA设备并设置环境变量
    try:
        # 尝试检测CUDA设备
        if paddle.is_compiled_with_cuda():
            device_count = paddle.device.cuda.device_count()
            if device_count > 0:
                print(f'[CUDA] 检测到 {device_count} 个CUDA设备，使用GPU训练')
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                force_cpu = False
            else:
                print('[CUDA] 未检测到可用的CUDA设备，强制使用CPU')
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                force_cpu = True
        else:
            print('[CUDA] PaddlePaddle未编译CUDA支持，使用CPU')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            force_cpu = True
    except Exception as e:
        print(f'[CUDA] 设备检测异常: {str(e)}，强制使用CPU')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        force_cpu = True
    
    # 设置PaddlePaddle设备
    try:
        if force_cpu:
            paddle.set_device('cpu')
            print('[设备] 已设置使用CPU进行训练')
        else:
            paddle.set_device('gpu:0')
            print('[设备] 已设置使用GPU进行训练')
    except Exception as e:
        print(f'[设备] 设置设备失败，回退到CPU: {str(e)}')
        paddle.set_device('cpu')
        force_cpu = True

    # sys.path.append( '/data/tensorflow/kath/Service/LegalDepartment/OCRProject/backend/TemplateOCR/Funcs/PaddleOCRrelease/PaddleOCRrelease')
    program=load_module_from(  path1+r'/TemplateOCR/Funcs/PaddleOCRrelease/PaddleOCRrelease/tools', 'program')
    # import tools.program as program
    import paddle.distributed as dist
    import inspect
    # print('[file source]',inspect.getfile(program))
    # print('[file source]',inspect.getfile(program.preprocess))
    dist.get_world_size()
    yml_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + r'/static/media/template_file/det.yml'
    
    try:
        config, device, logger, vdl_writer = program.preprocess(is_train=True,train_mode='det',config_path=yml_path,pretrained_dir=pretrained_dir,use_visualdl=True)
        
        # 如果强制使用CPU，确保配置中也设置为CPU
        if force_cpu:
            if 'Global' in config:
                config['Global']['use_gpu'] = False
            device = 'cpu'
            print('[配置] 已在配置文件中强制设置使用CPU')
            
    except Exception as e:
        print(f'[训练] program.preprocess调用失败: {str(e)}')
        # 如果还是失败，尝试完全强制CPU模式
        if not force_cpu:
            print('[训练] 尝试强制CPU模式重新初始化')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            paddle.set_device('cpu')
            try:
                config, device, logger, vdl_writer = program.preprocess(is_train=True,train_mode='det',config_path=yml_path,pretrained_dir=pretrained_dir,use_visualdl=True)
                if 'Global' in config:
                    config['Global']['use_gpu'] = False
                device = 'cpu'
                print('[配置] 强制CPU模式初始化成功')
            except Exception as e2:
                print(f'[训练] 强制CPU模式仍然失败: {str(e2)}')
                raise e2
        else:
            raise e
    # config['Global']['save_model_dir']=r'output\det_ct_tax'

    

    config['Global']['checkpoints'] = pretrained_dir
    config['Global']['epoch_num'] = checkpoint['epoch']+5
    config['Optimizer']['lr']['epochs'] = config['Global']['epoch_num']
    config['Global']['pretrained_model'] = pretrained_dir
    config['Train']['data_dir'] = dataPath
    config['Train']['label_file_list'] = [dataPath + r'/train.txt']
    config['Train']['dataset']['data_dir'] = dataPath
    config['Train']['dataset']['label_file_list'] = [dataPath + r'/train.txt']

    config['Eval']['data_dir'] = dataPath
    config['Eval']['label_file_list'] = [dataPath + r'/val.txt']
    config['Eval']['dataset']['data_dir'] = dataPath
    config['Eval']['dataset']['label_file_list'] = [dataPath + r'/val.txt']
    config['Metric']['main_indicator'] = 'recall'
    set_seed(2023)
    print('[DET] det model has been loaded in memory.')

    print()
    print('[DET] start training...(pass)')
    # return 0
    
    train_main=load_module_from(  path1+r'/TemplateOCR/Funcs/PaddleOCRrelease/PaddleOCRrelease/tools', 'train')
    
    update_status(0, status='running_det', progress=3, error='')
    train_main.main(config, device, logger, vdl_writer)
    print('[DET] det over')

    with open(yml_path,'w') as f:
      yaml.dump(config,f, default_flow_style=False, allow_unicode=True)




def SER_retrainer(pretrained_dir,dataPath,target_model_path):
    # Dataset_generator
    print('[SER] Loading ser model...')
    # 确保paddle为全局导入的模块
    import paddle
    checkpoint = paddle.load(os.path.join(pretrained_dir, 'metric.states'))
    checkpoint['best_model_dict']['hmean'] = 0.7
    checkpoint['best_model_dict']['recall'] = 0.7
    checkpoint['best_model_dict']['precision'] = 0.7
    checkpoint['hmean'] = 0.7
    checkpoint['precision'] = 0.7
    checkpoint['recall'] = 0.7

    paddle.save(checkpoint, os.path.join(pretrained_dir, 'metric.states'))

    # 检测CUDA设备并设置环境变量
    try:
        # 尝试检测CUDA设备
        if paddle.is_compiled_with_cuda():
            device_count = paddle.device.cuda.device_count()
            if device_count > 0:
                print(f'[CUDA-SER] 检测到 {device_count} 个CUDA设备，使用GPU训练')
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                force_cpu = False
            else:
                print('[CUDA-SER] 未检测到可用的CUDA设备，强制使用CPU')
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                force_cpu = True
        else:
            print('[CUDA-SER] PaddlePaddle未编译CUDA支持，使用CPU')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            force_cpu = True
    except Exception as e:
        print(f'[CUDA-SER] 设备检测异常: {str(e)}，强制使用CPU')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        force_cpu = True

    # 设置PaddlePaddle设备
    try:
        if force_cpu:
            paddle.set_device('cpu')
            print('[设备-SER] 已设置使用CPU进行训练')
        else:
            paddle.set_device('gpu:0')
            print('[设备-SER] 已设置使用GPU进行训练')
    except Exception as e:
        print(f'[设备-SER] 设置设备失败，回退到CPU: {str(e)}')
        paddle.set_device('cpu')
        force_cpu = True

    # import tools.train as trainer
    program = load_module_from(path1 + r'/TemplateOCR/Funcs/PaddleOCRrelease/PaddleOCRrelease/tools', 'program')
    import paddle.distributed as dist
    import inspect

    dist.get_world_size()
    yml_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + r'/static/media/template_file/ser.yml'

    try:
        config, device, logger, vdl_writer = program.preprocess(is_train=True, train_mode='ser', config_path=yml_path, pretrained_dir=pretrained_dir, use_visualdl=True)
        # 如果强制使用CPU，确保配置中也设置为CPU
        if force_cpu:
            if 'Global' in config:
                config['Global']['use_gpu'] = False
            device = 'cpu'
            print('[配置-SER] 已在配置文件中强制设置使用CPU')
    except Exception as e:
        print(f'[训练-SER] program.preprocess调用失败: {str(e)}')
        # 如果还是失败，尝试完全强制CPU模式
        if not force_cpu:
            print('[训练-SER] 尝试强制CPU模式重新初始化')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            paddle.set_device('cpu')
            try:
                config, device, logger, vdl_writer = program.preprocess(is_train=True, train_mode='ser', config_path=yml_path, pretrained_dir=pretrained_dir, use_visualdl=True)
                if 'Global' in config:
                    config['Global']['use_gpu'] = False
                device = 'cpu'
                print('[配置-SER] 强制CPU模式初始化成功')
            except Exception as e2:
                print(f'[训练-SER] 强制CPU模式仍然失败: {str(e2)}')
                raise e2
        else:
            raise e

    config['Global']['epoch_num'] = checkpoint['epoch'] + 5
    config['Optimizer']['lr']['epochs'] = config['Global']['epoch_num']
    config['Global']['save_model_dir'] = target_model_path  # r'output/ser_ct_tax'
    config['Global']['save_inference_dir'] = target_model_path

    config['Architecture']['Backbone']['pretrained'] = pretrained_dir
    config['Architecture']['Backbone']['checkpoints'] = pretrained_dir
    config['Train']['data_dir'] = dataPath
    config['Train']['label_file_list'] = [dataPath + r'/train.txt']
    config['Train']['ratio_list'] = [1]
    config['Train']['dataset']['data_dir'] = dataPath
    config['Train']['dataset']['label_file_list'] = [dataPath + r'/train.txt']
    config['Train']['dataset']['ratio_list'] = [1]
    config['PostProcess']['class_path'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + r'/static/media/template_file/allLabel_tax.txt'
    config['Train']['dataset']['transforms'][1]['VQATokenLabelEncode']['class_path'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + r'/static/media/template_file/allLabel_tax.txt'
    config['Global']['kie_rec_model_dir'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + r'TemplateOCR/Funcs/inferenced_models/rec'
    config['Global']['kie_det_model_dir'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + r'TemplateOCR/Funcs/inferenced_models/det'
    config['Global']['rec_model_dir'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + r'TemplateOCR/Funcs/inferenced_models/rec'
    config['Global']['det_model_dir'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + r'TemplateOCR/Funcs/inferenced_models/det'
    config['Eval']['data_dir'] = dataPath
    config['Eval']['label_file_list'] = [dataPath + r'/val.txt']
    config['Eval']['dataset']['data_dir'] = dataPath
    config['Eval']['dataset']['label_file_list'] = [dataPath + r'/val.txt']
    config['Eval']['dataset']['transforms'][1]['VQATokenLabelEncode']['class_path'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + r'/static/media/template_file/allLabel_tax.txt'

    set_seed(2023)
    print('[SER] det model has been loaded in memory.')

    print()
    print('[SER] start training...')
    train_main = load_module_from(path1 + r'/TemplateOCR/Funcs/PaddleOCRrelease/PaddleOCRrelease/tools', 'train')
    train_main.main(config, device, logger, vdl_writer)
    print('[SER] ending...')
    with open(yml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return 0




def trans_model2onnx(dataPath,ori_model_path, target_model_path):
    yml_path=yml_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + r'/static/media/template_file/det.yml'
    FLAGS = argparse.Namespace(config=yml_path, opt={}, profiler_options=None)
    config = export_model.load_config(FLAGS.config)
    config['Global']['save_model_dir'] = r'output/ser_ct_tax'
    config['Global']['checkpoints'] = ori_model_path
    config['Global']['save_inference_dir']=target_model_path
    config['Global']['pretrained_model']=ori_model_path
    config['Architecture']['Backbone']['pretrained'] = ori_model_path
    config['Train']['data_dir'] = dataPath
    config['Train']['label_file_list'] = [dataPath + r'/train.txt']
    config['Train']['dataset']['data_dir'] = dataPath
    config['Train']['dataset']['label_file_list'] = [dataPath + r'/train.txt']
    config['PostProcess']['class_path'] = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))) + r'/static/media/template_file/allLabel_tax.txt'

    config['Eval']['data_dir'] = dataPath
    config['Eval']['label_file_list'] = [dataPath + r'/val.txt']
    config['Eval']['dataset']['data_dir'] = dataPath
    config['Eval']['dataset']['label_file_list'] = [dataPath + r'/val.txt']
    # config['Eval']['dataset']['transforms'][1]['VQATokenLabelEncode']['class_path'] = os.path.dirname(
    #     os.path.dirname(os.path.abspath(__file__))) + r'\static\media\template_file\allLabel_tax.txt'
    # import tools.export_model as export_model
    program=load_module_from(  path1+r'/TemplateOCR/Funcs/PaddleOCRrelease/PaddleOCRrelease/tools', 'export_model')
    program.main(config)



    # export_model.main(config)
    # os.system('''paddle2onnx --model_dir "+target_model_path+ ' --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ' + os.path.dirname(target_model_path)+'/det_ct.onnx' + ' --opset_version 10 --input_shape_dict="{'x":[-1,3,-1,-1]}" --enable_onnx_checker True''')


    return 0

def move_ser_model2dir():

    shutil.copy(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))) + r'\static\media\template_file\allLabel_tax.txt',
                r'TemplateOCR\Funcs\inferenced_models\ser')
    shutil.move(
        r'TemplateOCR\Funcs\inferenced_models\ser\allLabel_tax.txt',
                r'TemplateOCR\Funcs\inferenced_models\ser\allLabel.txt')
    shutil.move(
        r'output\ser_ct_tax\best_model\model.pdopt',
        r'TemplateOCR\Funcs\inferenced_models\ser\model.pdopt')
    shutil.move(
        r'output\ser_ct_tax\best_model\model_config.json',
        r'TemplateOCR\Funcs\inferenced_models\ser\model_config.json')
    shutil.move(
        r'output\ser_ct_tax\best_model\model_state.pdparams',
        r'TemplateOCR\Funcs\inferenced_models\ser\model_state.pdparams')

def readFile(path):
	with open(path,'r') as f:
	    res=[]
	    for line in f:
	    	res.append(line)
	return res

def readLabelFile_withoutCSV(label_path):
    dct={}
    with open(label_path,'r') as f:
        res=[]
        for line in f:
            temp=line.split('\t')
            NAME=temp[0]
            filename1=temp[0].split('.')
            filename=filename1[0]
            temp=temp[1]
            temp=re.sub('false','False',temp)
            temp=re.sub('true','True',temp)
            lst=eval(temp)

            lst=[x for x in lst if x['transcription']!='待识别']
            dct[NAME]=lst

            lst=json.dumps( lst ,ensure_ascii=False)

            res.append(NAME+'\t'+lst)

    return res,dct


# -------------------------------DATA AUGUMENTATION --------------------------------------
def sp_noise(image,prob):
    output=copy.deepcopy(image)
    thres = 1 - prob
    for i in range(0,image.shape[0],2):
        for j in range(0,image.shape[1],2):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 125
            elif rdn > thres:
                output[i][j] = 250
            else:
                output[i][j] = image[i][j]
    return output

def gasuss_noise(image, mean=0, var=0.001):
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out
def random_colormap(img_a,n):
    if n==0:
        img_a=cv2.applyColorMap(img_a, colormap=cv2.COLORMAP_PINK)
    if n==1:
        img_a=cv2.applyColorMap(img_a, colormap=cv2.COLORMAP_CIVIDIS)
    if n==2:
        #         img_a=cv2.applyColorMap(img_a, colormap=cv2.COLORMAP_HOT)
        img_a=cv2.applyColorMap(img_a, cmapy.cmap('BuPu_r'))
    if n==3:
        #         img_a=cv2.applyColorMap(img_a, colormap=cv2.COLORMAP_OCEAN)
        img_a=cv2.applyColorMap(img_a, cmapy.cmap('bone'))
    if n==4:
        img_a=cv2.applyColorMap(img_a, colormap=cv2.COLORMAP_DEEPGREEN)

    return img_a
def contrast_brightness(img,num):
    res=[]
    bri_mean = np.mean(img)

    a = np.arange(7, 20, 3) / 10
    #     b = np.arange(-30, 21, 10)
    b = np.arange(-40, 15, 10)

    for i in range(num):
        ra=random.randint(0,5)
        rb=random.randint(0,6)
        if ra>3 or rb>5:
            res.append(img)
            continue
        aa=a[ra]
        bb=b[rb]
        img_a = aa * (img-bri_mean) + bb + bri_mean
        img_a = np.clip(img_a,0,255).astype(np.uint8)
        seed=random.randint(0,12)
        img_a=random_colormap(img_a,seed)
        #         img_show(img_a)
        res.append(img_a)
    #     print(len(res))
    return res


def angle_augment(img):
    ang=random.randint(-7,7)
    hh,ww=img.shape[0],img.shape[1]
    # if ang>-2 and ang<2:
    #     if ang<0:
    #         img_,M=img_rotate(img, -90,hh,ww)
    #         ang=-90
    #     else:
    #         img_,M=img_rotate(img, 90,hh,ww)
    #         ang=90
    # else:
    #     img_,M=img_rotate(img, ang,hh,ww)
    img_, M = img_rotate(img, ang, hh, ww)
    return img_,ang,M

def img_rotate(src, angel,hh,ww):
    h,w = src.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angel, 1.0)
    rotated_h = int((w * np.abs(M[0,1]) + (h * np.abs(M[0,0]))))
    rotated_w = int((h * np.abs(M[0,1]) + (w * np.abs(M[0,0]))))

    M[0,2] += (rotated_w - w) // 2
    M[1,2] += (rotated_h - h) // 2

    if int(rotated_w*(hh/ww))>rotated_h:
        rotated_img = cv2.warpAffine(src, M, (rotated_w,int(rotated_w*(hh/ww))),borderValue=(255,255,255))
    else:
        rotated_img = cv2.warpAffine(src, M, ( int(rotated_h*(ww/hh))  ,rotated_h ),borderValue=(255,255,255))

    return rotated_img,M
def resize_boxes_cal(scal,infor_data):
    res=[]
    for item in infor_data:
        temp=copy.deepcopy(item)
        temp['points']=[[int(i[0]*scal),int(i[1]*scal) ] for i in item['points']]
        res.append(temp)
    return res
def rotate(ps,m):
    pts = np.float32(ps).reshape([-1, 2])  # 要映射的点
    pts = np.hstack([pts, np.ones([len(pts), 1])]).T
    target_point = np.dot(m, pts)
    target_point = [[int(target_point[0][x]),int(target_point[1][x])] for x in range(len(target_point[0]))]
    return target_point

def rotate_img_and_point(M,points,angle):
    out_points = rotate(points,M)
    return out_points
def reangle_boxes_cal(ang,infor_data,M):
    res=[]
    for item in infor_data:
        temp=copy.deepcopy(item)
        temp['points']=rotate_img_and_point(M,item['points'],ang)
        res.append(temp)
    return res

def rotation(img,num):
    imgs=[]
    scales=[]
    angles=[]
    Ms=[]
    for i in range(num):

        res,angle,M=angle_augment(img)
        angles.append(angle)
        Ms.append(M)
        imgs.append(res)
    return imgs,scales,angles,Ms

def rotation_imgs_info(infor_data,im,num):
    imgs,scales,angles,Ms=rotation(im,num)
    res_infos=[]
    for img,angle,M in zip(imgs,angles,Ms):
        reangle_info=reangle_boxes_cal(angle,infor_data,M)
        res_infos.append(reangle_info)

    return imgs,res_infos

def map_x_to_two_ints(x):
    if x >= 150:
        return 1,1
    if x<=2:
        return 6,8
    
    min_prod = math.ceil(50 / x)
    max_prod = math.floor(150 / x)
    
    if min_prod > max_prod:
        raise ValueError(f"No valid (a,b) possible for x={x}")

    # 选取中间值作为目标乘积
    target = (min_prod + max_prod) // 2

    # 分解 target 为两个整数 a, b 使得 a * b ≈ target
    a = int(math.sqrt(target))
    b = round(target / a)

    # 修正乘积不合法的情况
    product = a * b
    if product < min_prod:
        b = math.ceil(min_prod / a)
    elif product > max_prod:
        b = max(1, int(max_prod / a))

    # 再检查结果合法性
    final_product = a * b * x
    if not (50 <= final_product <= 150):
        raise ValueError(f"Failed to satisfy condition: a={a}, b={b}, x={x}, a*b*x={final_product}")

    return a, b


def data_augumentation(path,target_img_path,Labels_dct):
    print('Labels_dct',Labels_dct)
    aug_ctl=len(Labels_dct)
    aug_ctl1,aug_ctl2= map_x_to_two_ints(aug_ctl)
    

    to_txtFile=[]
    count=0
    for file in os.listdir(path):   #+'img\\'
        if 'jpg' not in file and  'png' not in file and  'jpeg' not in file:
            continue
        print(file,count)
        count=count+1
        if file not in Labels_dct:
            print('cannot find '+file+' in label file')
            continue
        infor_data=Labels_dct[file]
        img=eval("cv2.imread(r'"+path+file+"\' )")
        cv2.imwrite(target_img_path +file ,img)
        to_txtFile.append( file+'\t'+json.dumps(infor_data ,ensure_ascii=False))
        imgs,info1=rotation_imgs_info(infor_data,img,aug_ctl1) #6

        ct=0
        subct=0
        for im in imgs:
            #             t3=time.time()
            if max(im.shape)>1600:
                temp=random.randint(0,2)
                if temp<2:
                    prob=random.randint(0,10)*0.01
                    noise = sp_noise(im,prob)  # 高斯低通滤波处理
                else:
                    noise = im
                temp=random.randint(0,2)

                if temp<2 :
                    mean=random.randint(-5,10)*0.01
                    var=random.randint(0,20)*0.001
                    res = gasuss_noise(noise, mean, var)
                else:
                    res=noise
            else:
                res=im
            if max(res.shape)>2500:
                resi=random.randint(46,76)*0.01

            else:
                resi=random.randint(90,100)*0.01

            res=cv2.resize(res,(int(res.shape[1]*resi),int(res.shape[0]*resi)))
            resi_boxes = resize_boxes_cal(resi,info1[subct])
            subct=subct+1

            cbs=contrast_brightness(res,aug_ctl2) #8
            for m in cbs:
                ct=ct+1
                to_txtFile.append(file.split('.')[0]+'_'+str(ct)+'.jpg'+'\t'+json.dumps(resi_boxes ,ensure_ascii=False))

                temppath=target_img_path +file.split('.')[0]+'_'+str(ct)+'.jpg'

                eval("cv2.imwrite(r'"+temppath+"\' ,m)")

    #         break
    return to_txtFile


def generat_class_file():
    # -----------generate trainingset file------------
    print('generating trainset')
    ori_data_path=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + r'/data/Dataset_small/-train.txt'
    res_ori_,dct_ori=readLabelFile_withoutCSV(ori_data_path)
    res_ori=[]
    for i in res_ori_:
        x=random.randint(-2, 6)
        if x>=0:
            res_ori.append(i)
    new_data_path=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))) + r'/static/media/template_file/Label.txt'
    res_new, dct_new = readLabelFile_withoutCSV(new_data_path)
    res=copy.deepcopy(res_ori)
    dct_new1={}
    #删除Dataset_new原有元素
    # file_path = os.path.dirname(os.path.abspath(__file__)) + r'\data\Dataset_small\Dataset_new'
    # fs = os.listdir(file_path)
    # try:
    #     for m in fs:
    #         os.remove(file_path + m)
    # except OSError as e:
    #     print('cannot delete file', e)

    img_lst = os.listdir(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + r'/data/Dataset_small/Dataset_new/')
    for item in img_lst:
        os.remove(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + r'/data/Dataset_small/Dataset_new/'+item)
    new_class =[]
    for img in dct_new:
        print('[debuhg]',os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        shutil.copy(img,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + r'/data/Dataset_small/Dataset_new/')
        # print('[   check   ]',img,os.path.basename(img))
        dct_new1[os.path.basename(img)]=dct_new[img]
        for i in dct_new[img]:
            new_class.append(i['label']+'\n')
    new_class=list(set(new_class))
    print(' data augumentation...')
    to_textFile=data_augumentation(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + r'/data/Dataset_small/Dataset_new/',
                                   os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + r'/data/Dataset_small/Dataset_new/',
                                   dct_new1) 

    with open(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + r'/data/Dataset_small/val.txt', 'w') as f:
      for it in to_textFile:
          
          temp='Dataset_new/'+it.split('\t')[0].split('/')[-1]+'\t'+it.split('\t')[-1]
          
          if random.randint(0,10) >7:
            f.write(temp + '\n')
          else:
            res.append(temp)

    with open(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + r'/data/Dataset_small/val.txt', 'a') as ff:
      
      with open(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + r'/data/Dataset_small/train.txt', 'w') as f:
          for i in res:
              i=re.sub('TITLE','title' , i)
              
              if random.randint(0,20) >=18:
                ff.write(i + '\n')
              else:
                f.write(i + '\n')

    # with open(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + r'/data/Dataset_small/val.txt', 'w') as f:
    #     for i in res_new:
    #         i=re.sub('TITLE','title' , i)
    #         i='Dataset_new/'+os.path.basename(i.split('\t')[0])+'\t'+i.split('\t')[-1]
    #         f.write(i + '\n')
    # -----------generate class file------------
    ori_class=readFile(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + r'/static/media/template_file/allLabel_tax.txt')
    res_class=copy.deepcopy(ori_class)
    new_class=[i for i in new_class if i not in ori_class]
    # print('new_class:',new_class)
    count=0
    for ind, it in enumerate(ori_class):
        if it[:-1].isnumeric() and count<len(new_class):
            res_class[ind]=new_class[count]
            print('newadded'+new_class[count])
            count=count+1
    # print('*******newclass',res_class[-1002:-990])
    with open(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + r'/static/media/template_file/allLabel_tax.txt', 'w') as f:
        for i in res_class:
            f.write(i )

    print('generat_class_file DONE')



    return 0







if __name__ == '__main__':
    generat_class_file()
    # r"D:\trans_work_jian\suibian\OCRProject\OCRProject\backend\static\media\template_file\rec.yml"
    # recModel=REC_Model()
    # recognization(r"D:\trans_work_jian\suibian\OCRProject\OCRProject\backend\static\media\template_file\Label.txt",recModel)
    # print()
    # DET_retrainer(r"D:\trans_work_jian\suibian\OCRProject\OCRProject\backend\TemplateOCR\Funcs\inferenced_models\det\pretrained\model",r'D:\trans_work_jian\suibian\OCRProject\OCRProject\data\Dataset_small')
    # SER_retrainer(
    #     r"D:\trans_work_jian\suibian\OCRProject\OCRProject\backend\TemplateOCR\Funcs\inferenced_models\ser\inference",
    #     r'D:\trans_work_jian\suibian\OCRProject\OCRProject\data\Dataset_small',
    # r'D:\trans_work_jian\suibian\OCRProject\OCRProject\backend\TemplateOCR\Funcs\inferenced_models\det2')

    # trans_model2onnx(r'D:\trans_work_jian\suibian\OCRProject\OCRProject\data\Dataset_small',
    #                  r'D:\trans_work_jian\suibian\OCRProject\OCRProject\backend\output\det_ct_tax\best_model\model',
    #                  r'D:\trans_work_jian\suibian\OCRProject\OCRProject\backend\TemplateOCR\Funcs\inferenced_models\det2')
    # move_ser_model2dir()
    print('done!')


