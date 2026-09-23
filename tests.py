# from django.test import TestCase

# Create your tests here.

import pandas as pd
from paddleocr.paddleocr import PaddleOCR, draw_ocr
import matplotlib.pyplot as plt
import os
import distance
import re
import cv2
import copy
import numpy as np
import argparse
import time



ocr=PaddleOCR(use_gpu=False, use_xpu=False, use_npu=False, ir_optim=True, use_tensorrt=False, min_subgraph_size=15, precision='fp32', gpu_mem=500, gpu_id=0, image_dir='D:\\trans_work_jian\\suibian\\OCRProject\\OCRProject\\backend\\static\\media\\template_img\\\\zh_val_39.jpg', page_num=0,
              det_algorithm='DB', det_model_dir='D:\\trans_work_jian\\suibian\\OCRProject\\OCRProject\\backend\\TemplateOCR\\Funcs\\inferenced_models\\det_ct.onnx', det_limit_side_len=960, det_limit_type='max', det_box_type='quad',
                       det_db_thresh=0.3, det_db_box_thresh=0.6, det_db_unclip_ratio=1.5, max_batch_size=10, use_dilation=False, det_db_score_mode='fast', det_east_score_thresh=0.8, det_east_cover_thresh=0.1, det_east_nms_thresh=0.2, det_sast_score_thresh=0.5, det_sast_nms_thresh=0.2, det_pse_thresh=0, det_pse_box_thresh=0.85, det_pse_min_area=16, det_pse_scale=1, scales=[8, 16, 32], alpha=1.0, beta=1.0, fourier_degree=5, rec_algorithm='SVTR_LCNet', rec_model_dir='D:\\trans_work_jian\\suibian\\OCRProject\\OCRProject\\backend\\TemplateOCR\\Funcs\\inferenced_models\\rec_v4.onnx', rec_image_inverse=True, rec_image_shape='3, 48, 320', rec_batch_num=6, max_text_length=25, rec_char_dict_path='D:\\trans_work_jian\\suibian\\OCRProject\\OCRProject\\backend\\TemplateOCR\\Funcs\\PaddleOCRrelease\\PaddleOCRrelease\\ppocr\\utils\\ppocr_keys_v1.txt', use_space_char=True, vis_font_path='./doc/fonts/simfang.ttf', drop_score=0.5, e2e_algorithm='PGNet', e2e_model_dir=None, e2e_limit_side_len=768, e2e_limit_type='max', e2e_pgnet_score_thresh=0.5, e2e_char_dict_path='./ppocr/utils/ic15_dict.txt', e2e_pgnet_valid_set='totaltext', e2e_pgnet_mode='fast', use_angle_cls=False, cls_model_dir='D:\\trans_work_jian\\suibian\\OCRProject\\OCRProject\\backend\\TemplateOCR\\Funcs\\inferenced_models\\cls.onnx', cls_image_shape='3, 48, 192', label_list=['0', '180'], cls_batch_num=6, cls_thresh=0.9, enable_mkldnn=False, cpu_threads=10, use_pdserving=False, warmup=False, sr_model_dir=None, sr_image_shape='3, 32, 128', sr_batch_num=1, draw_img_save_dir='./inference_results', save_crop_res=False, crop_res_save_dir='./output', use_mp=False, total_process_num=1, process_id=0, benchmark=False, save_log_path='./log_output/', show_log=True, use_onnx=True)

img=cv2.imread(r"D:\kath-workfile\Data\营业执照\dataset\ori_businesslicense\businesslicense_00001807.jpg")
t1=time.time()
result=ocr.ocr(img)
t2=time.time()
print('predict time :',t2-t1)
    #     (
    # det_db_box_thresh=0.3,
    # #det_db_unclip_ratio=1.6,
    # )
from PIL import Image
result = result[0]
# image = Image.open(img).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(img, boxes, txts, scores, font_path='./fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
print()
# im_show.save('./0res'+docName+'result.jpg')