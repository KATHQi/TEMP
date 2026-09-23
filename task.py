from time import sleep
from celery import shared_task
from django.http import JsonResponse
from django.shortcuts import HttpResponse
import os,sys


from functions import *

@shared_task
def async_task(pretrained_dir,dataPath):

    DET_retrainer(pretrained_dir,dataPath)
    # if pretrained_dir!='':
    #     print(os.path.dirname(os.path.abspath(__file__)))
    #     os.sys(r'python '+os.path.dirname(os.path.abspath(__file__))+r'/ab.py')
    # generat_class_file()
    # DET_retrainer(
    #     r"D:\trans_work_jian\suibian\OCRProject\OCRProject\backend\TemplateOCR\Funcs\inferenced_models\det\pretrained\model",
    #     r'D:\trans_work_jian\suibian\OCRProject\OCRProject\data\Dataset_small')
    # SER_retrainer(
    #     r"D:\trans_work_jian\suibian\OCRProject\OCRProject\backend\TemplateOCR\Funcs\inferenced_models\ser\inference",
    #     r'D:\trans_work_jian\suibian\OCRProject\OCRProject\data\Dataset_small')
    # trans_model2onnx(r'D:\trans_work_jian\suibian\OCRProject\OCRProject\data\Dataset_small',
    #                  r'D:\trans_work_jian\suibian\OCRProject\OCRProject\backend\output\det_ct_tax\best_model\model',#det与SER不一样！！！用latest
    #                  r'D:\trans_work_jian\suibian\OCRProject\OCRProject\backend\TemplateOCR\Funcs\inferenced_models\det2')
    # move_ser_model2dir()
    #
    return 'Done1'

# @shared_task
# def async_task(x,y):
#     res=0
#     for i in range(5):
#         sleep(1)
#         print('aaa:'+str(x+y+i))
#         res=x+y+i
#
#     return 'Done'




