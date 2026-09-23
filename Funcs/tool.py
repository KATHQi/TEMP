# import subprocess

# command='python ./PaddleOCR-release-2.7/PaddleOCR-release-2.7/tools/infer/predict_det.py --det_algorithm="SAST" --det_model_dir='
import time
import sys,os
import cv2
import re

# sys.path.append( path1+r'\TemplateOCR\Funcs\PaddleOCRrelease\PaddleOCRrelease')
path1=os.path.dirname(os.path.abspath(__file__))
sys.path.append( path1+r'\PaddleOCRrelease\PaddleOCRrelease')
# import
import tools.infer.utility as utility
# print(path1+r'\TemplateOCR\Funcs\PaddleOCRrelease')
# from PaddleOCRrelease.PaddleOCRrelease.tools.infer.predict_system import *
from tools.infer.predict_system import *

# from TemplateOCR.Funcs.PaddleOCRrelease.PaddleOCRrelease.tools.infer.predict_system import *
# from tools.infer.predict_det import *
# from Funcs.PaddleOCRrelease.PaddleOCRrelease.tools.infer_kie_token_ser import *
# from PaddleOCRrelease.PaddleOCRrelease.tools.infer_kie_token_ser import *
from tools.infer_kie_token_ser import *
#
# from Funcs.PaddleOCRrelease.PaddleOCRrelease.ppstructure.kie.predict_kie_token_ser import SerPredictor
# from ppstructure.kie.predict_kie_token_ser import SerPredictor
# from ppocr.utils.utility import get_image_file_list, check_and_read
import tools.program as program
from PIL import Image





import argparse
my_arg=argparse.Namespace(use_gpu=False, use_xpu=False, use_npu=False, ir_optim=True, use_tensorrt=False, min_subgraph_size=15, precision='fp32', gpu_mem=500, gpu_id=0, image_dir='', page_num=0, det_algorithm='DB', det_model_dir=path1+'\\inferenced_models\\det_ct.onnx', det_limit_side_len=960, det_limit_type='max', det_box_type='quad', det_db_thresh=0.3, det_db_box_thresh=0.6, det_db_unclip_ratio=1.5, max_batch_size=10, use_dilation=False, det_db_score_mode='fast', det_east_score_thresh=0.8, det_east_cover_thresh=0.1, det_east_nms_thresh=0.2, det_sast_score_thresh=0.5, det_sast_nms_thresh=0.2, det_pse_thresh=0, det_pse_box_thresh=0.85, det_pse_min_area=16, det_pse_scale=1, scales=[8, 16, 32], alpha=1.0, beta=1.0, fourier_degree=5, rec_algorithm='SVTR_LCNet', rec_model_dir=path1+'\\inferenced_models\\rec_v4.onnx', rec_image_inverse=True, rec_image_shape='3, 48, 320', rec_batch_num=6, max_text_length=25, rec_char_dict_path='./ppocr/utils/ppocr_keys_v1.txt', use_space_char=True, vis_font_path='./doc/fonts/simfang.ttf', drop_score=0.5,
e2e_algorithm='PGNet', e2e_model_dir=None, e2e_limit_side_len=768, e2e_limit_type='max', e2e_pgnet_score_thresh=0.5, e2e_char_dict_path='./ppocr/utils/ic15_dict.txt', e2e_pgnet_valid_set='totaltext', e2e_pgnet_mode='fast', use_angle_cls=False, cls_model_dir=None, cls_image_shape='3, 48, 192', label_list=['0', '180'], cls_batch_num=6, cls_thresh=0.9, enable_mkldnn=False, cpu_threads=10, use_pdserving=False, warmup=False, sr_model_dir=None, sr_image_shape='3, 32, 128', sr_batch_num=1, draw_img_save_dir='./inference_results', save_crop_res=False, crop_res_save_dir='./output', use_mp=False, total_process_num=1, process_id=0, benchmark=False, save_log_path='./log_output/', show_log=True, use_onnx=True)


kie_arg=argparse.Namespace(alpha=1.0, alphacolor=(255, 255, 255), benchmark=False, beta=1.0, binarize=False, cls_batch_num=6, cls_image_shape='3, 48, 192', cls_model_dir='/data/kath/models/ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer', cls_thresh=0.9, cpu_threads=10, crop_res_save_dir='./output', det_algorithm='DB', det_box_type='quad', det_db_box_thresh=0.6, det_db_score_mode='fast', det_db_thresh=0.3, det_db_unclip_ratio=1.5, det_east_cover_thresh=0.1, det_east_nms_thresh=0.2, det_east_score_thresh=0.8, det_limit_side_len=960, det_limit_type='max', det_model_dir='', det_pse_box_thresh=0.85, det_pse_min_area=16, det_pse_scale=1, det_pse_thresh=0, det_sast_nms_thresh=0.2, det_sast_score_thresh=0.5, draw_img_save_dir='./inference_results', drop_score=0.5, e2e_algorithm='PGNet', e2e_char_dict_path='./ppocr/utils/ic15_dict.txt', e2e_limit_side_len=768, e2e_limit_type='max', e2e_model_dir=None, e2e_pgnet_mode='fast', e2e_pgnet_score_thresh=0.5, e2e_pgnet_valid_set='totaltext', enable_mkldnn=False, fourier_degree=5, gpu_id=0, gpu_mem=500, image_dir='/data/kath/sources/examples', image_orientation=False, invert=False, ir_optim=True, kie_algorithm='LayoutXLM', label_list=['0', '180'], layout=True, layout_dict_path='../ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt', layout_model_dir=None, layout_nms_threshold=0.5, layout_score_threshold=0.5, max_batch_size=10, max_text_length=25, merge_no_span_structure=True, min_subgraph_size=15, mode='structure', ocr=True, ocr_order_method='tb-yx', output='./output', page_num=0, precision='fp32', process_id=0, re_model_dir=None, rec_algorithm='SVTR_LCNet', rec_batch_num=6, rec_char_dict_path='./ppocr/utils/ppocr_keys_v1.txt', rec_image_inverse=True, rec_image_shape='3, 48, 320', rec_model_dir='/data/kath/models/ch_PP-OCRv4_rec_infer', recovery=False, save_crop_res=False, save_log_path='./log_output/', scales=[8, 16, 32], ser_dict_path='/data/kath/data/Datasets/allLabel.txt', ser_model_dir='../inferencedModel/ser_vi_layout_infer', show_log=True, sr_batch_num=1, sr_image_shape='3, 32, 128', sr_model_dir=None, table=True, table_algorithm='TableAttn', table_char_dict_path='../ppocr/utils/dict/table_structure_dict_ch.txt', table_max_len=488, table_model_dir=None, total_process_num=1, use_angle_cls=False, use_dilation=False, use_gpu=True, use_mp=False, use_npu=False, use_onnx=False, use_pdf2docx_api=False, use_pdserving=False, use_space_char=True, use_tensorrt=False, use_visual_backbone=True, use_xpu=False, vis_font_path='../doc/fonts/simfang.ttf', warmup=False)


#
def text_pred_res_det(img,image_file,savename):

    args = my_arg#utility.parse_args()
    t1=time.time()
    print('========================')
    args.use_gpu=True
    args.use_onnx=True
    args.det_model_dir=path1+r'\inferenced_models\det_ct.onnx'
    args.rec_model_dir=path1+r'\inferenced_models\rec_v4.onnx'
    args.cls_model_dir=path1+r'\inferenced_models\cls.onnx'
    args.image_dir= image_file
    args.rec_char_dict_path=path1+r'\PaddleOCRrelease\PaddleOCRrelease\ppocr\utils\ppocr_keys_v1.txt'
    # args.det_algorithm='CT'
    args.rec_algorithm=''
    print(args)


    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list[args.process_id::args.total_process_num]
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    drop_score = args.drop_score
    draw_img_save_dir = args.draw_img_save_dir
    os.makedirs(draw_img_save_dir, exist_ok=True)
    save_results = []
    total_time = 0
    cpu_mem, gpu_mem, gpu_util = 0, 0, 0
    _st = time.time()
    count = 0

    starttime=time.time()
    if img is None:
        print('error in loading image')
        return


    dt_boxes, rec_res, time_dict = text_sys(img)
    elapse = time.time() - starttime
    total_time += elapse

    for text, score in rec_res:
        logger.debug("{}, {:.3f}".format(text, score))

    res = [{
        "transcription": rec_res[i][0],
        "points": np.array(dt_boxes[i]).astype(np.int32).tolist(),
        "label":'others'
    } for i in range(len(dt_boxes))]
    print('前期结果：', res)
    save_pred = image_file + "\t" + json.dumps(
        res, ensure_ascii=False) + "\n"
    save_results.append(save_pred)
    name=str(len(os.listdir(path1+"/outputs")))
    with open(
            os.path.join( path1, path1+"/outputs//"+name+'.txt'),
            'w',
            encoding='utf-8') as f:
        f.writelines(save_results)
    print('===results saved in txt file===')
    print( path1+"\outputs\\"+name+'.txt')
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    boxes = dt_boxes
    txts = [rec_res[i][0] for i in range(len(rec_res))]
    scores = [rec_res[i][1] for i in range(len(rec_res))]

    draw_img = utility.draw_text_det_res(boxes , img)
    t2 = time.time()
    print('DET+REC time:::', t2 - t1)
    # -----------------加入SER---------
    # SERCONF={'Global':
    #              {'use_gpu': False, 'epoch_num': 200, 'log_smooth_window': 10, 'print_batch_step': 10,
    #               'save_model_dir': './output/BigModel1', 'save_epoch_step': 2000, 'eval_batch_step': [0, 59],
    #               'cal_metric_during_train': False, 'save_inference_dir': None, 'use_visualdl': False,
    #               'seed': 2022, 'infer_img': path1+"/outputs//"+name+'.txt',
    #               'save_res_path': path1+"\\outputs\\serRes", 'kie_rec_model_dir':path1+ '\\inferenced_models\\ch_PP-OCRv4_rec_infer',
    #               'kie_det_model_dir': path1+'\\inferenced_models\\det', 'infer_mode': False,
    #               'det_algorithm': 'SAST', 'det_model_dir': path1+'\\inferenced_models\\det',
    #               'rec_model_dir': path1+'\\inferenced_models\\ch_PP-OCRv4_rec_infer',
    #               'cls_model_dir': path1+'\\inferenced_models\\cls', 'kie_cls_model_dir': path1+'\\inferenced_models\\cls',
    #               'class_path': path1+'\\inferenced_models\\ser\\allLabel.txt',
    #               'distributed': False},
    #          'Architecture': {'model_type': 'kie', 'algorithm': 'LayoutXLM', 'Transform': None,
    #                           'Backbone': {'name': 'LayoutXLMForSer', 'pretrained': path1+'\\inferenced_models\\ser\\inference',
    #                                        'checkpoints': path1+'\\inferenced_models\\ser\\inference',
    #                                        'mode': 'vi', 'num_classes': 8967}},
    #          'Loss': {'name': 'VQASerTokenLayoutLMLoss', 'num_classes': 8967, 'key': 'backbone_out'},
    #          'Optimizer': {'name': 'AdamW', 'beta1': 0.9, 'beta2': 0.999,
    #                        'lr': {'name': 'Linear', 'learning_rate': 5e-05, 'epochs': 200, 'warmup_epoch': 1},
    #                        'regularizer': {'name': 'L2', 'factor': 0.0}},
    #          'PostProcess': {'name': 'VQASerTokenLayoutLMPostProcess', 'class_path': path1+'\\inferenced_models\\ser\\allLabel.txt'},
    #          'Metric': {'name': 'VQASerTokenMetric', 'main_indicator': 'hmean'},
    #          'Train': {'dataset': {'name': 'SimpleDataSet', 'data_dir': '\\data/kath/data/Datasets/', 'label_file_list': ['/data/kath/data/Datasets/train.txt'], 'ratio_list': [1.0],
    #                                'transforms': [{'DecodeImage': {'img_mode': 'RGB', 'channel_first': False}}, {'VQATokenLabelEncode': {'contains_re': False, 'algorithm': 'LayoutXLM', 'class_path': path1+'\\inferenced_models\\ser\\allLabel.txt', 'use_textline_bbox_info': True, 'order_method': 'tb-yx'}}, {'VQATokenPad': {'max_seq_len': 512, 'return_attention_mask': True}}, {'VQASerTokenChunk': {'max_seq_len': 512}}, {'Resize': {'size': [224, 224]}}, {'NormalizeImage': {'scale': 1, 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'order': 'hwc'}}, {'ToCHWImage': None}, {'KeepKeys': {'keep_keys': ['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'image', 'labels']}}]},
    #                    'loader': {'shuffle': True, 'drop_last': False, 'batch_size_per_card': 16, 'num_workers': 0}}, 'Eval': {'dataset': {'name': 'SimpleDataSet', 'data_dir': '/data/kath/data/Datasets/', 'label_file_list': ['/data/kath/data/Datasets/val.txt'], 'transforms': [{'DecodeImage': {'img_mode': 'RGB', 'channel_first': False}}, {'VQATokenLabelEncode': {'contains_re': False, 'algorithm': 'LayoutXLM', 'class_path': path1+'\\inferenced_models\\ser\\SWallLabel.txt', 'use_textline_bbox_info': True, 'order_method': 'tb-yx'}}, {'VQATokenPad': {'max_seq_len': 512, 'return_attention_mask': True}}, {'VQASerTokenChunk': {'max_seq_len': 512}}, {'Resize': {'size': [224, 224]}},
    #                                 {'NormalizeImage': {'scale': 0.8, 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'order': 'hwc'}}, {'ToCHWImage': None}, {'KeepKeys': {'keep_keys': ['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'image', 'labels']}}]},
    #                                                                                                                            'loader': {'shuffle': False, 'drop_last': False, 'batch_size_per_card': 16, 'num_workers': 0}}, 'profiler_options': None}
    # ser_engine = SerPredictor(SERCONF)
    #
    # # infer_imgs = get_image_file_list(SERCONF['Global']['infer_img'])
    # infer_imgs=''
    # with open(SERCONF['Global']['infer_img'], "rb") as f:
    #     infer_imgs = f.readlines()
    # res = []
    # result=[]
    # for idx, info in enumerate(infer_imgs):
    #     if SERCONF["Global"].get("infer_mode", None) is False:
    #         data_line = info.decode('utf-8')
    #         substr = data_line.strip("\n").split("\t")
    #         img_path = substr[0]
    #         data = {'img_path': img_path, 'label': substr[1]}
    #     else:
    #         img_path = info
    #         data = {'img_path': img_path}
    #
    #     save_img_path = os.path.join(
    #         SERCONF['Global']['save_res_path'],
    #         os.path.splitext(os.path.basename(img_path))[0] + "_ser.jpg")
    #
    #
    #     result, _ = ser_engine(data)
    #     result = result[0]
    #     res.append(result)
    #     savename=re.sub('[: ]','_', savename)
    #     path2=os.path.dirname(os.path.dirname(path1))+r'\static\media\template_file\\'
    #     with open(path2+savename + '_ser.json', 'w') as file:
    #         file.write(os.path.dirname(os.path.dirname(path1))+'\\'+data['img_path'].split('\\')[-1] + '\t'+json.dumps(res[0], ensure_ascii=False))
    #         # for i in res[0]:
    #         #     file.write(json.dumps(i, ensure_ascii=False)+,)
    #         file.close()
    #     # fout.write(img_path + "\t" + json.dumps(
    #     #     {
    #     #         "ocr_info": result,
    #     #     }, ensure_ascii=False) + "\n")
    #     boxes = []
    #     for i in result:
    #         boxes.append(i['points'])
    #
    #
    #     img_res = draw_boxes_with_info(img, result)
    #     cv2.imwrite(save_img_path, img_res)
    #
    #     print("process: [{}/{}], save result to {}".format(
    #         idx, len(infer_imgs), save_img_path))
    # t_end = time.time()
    # print('total time:', t_end - t1)
    # print('ser(rec,cls,det loadingtime added) pred time:', t_end - t2)
    #
    # # ------SER ends----------
    #
    # save_file = image_file
    # # cv2.imwrite(
    # #     os.path.join(draw_img_save_dir,
    # #                  os.path.basename(save_file)),
    # #     draw_img[:, :, ::-1])
    # logger.debug("The visualized image saved in {}".format(
    #     os.path.join(draw_img_save_dir, os.path.basename(
    #         save_file))))
    #
    # # print(save_results)
    # print('SER res:',result)

    # return result,draw_img,savename + '_ser.json'#[:, :, ::-1]
    return 0


def draw_boxes(image, boxes):
    for box in boxes:
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)#
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    return image

def draw_boxes_with_info(im,infor_data):
    boxes=[]
    for i in infor_data:
        boxes.append(i['points'])
    res=draw_boxes(im, boxes)
    return res


INFERED_CONF={'Global':
                  {'use_gpu': False, 'epoch_num': 200, 'log_smooth_window': 10, 'print_batch_step': 10, 'save_model_dir': './output/BigModel1', 'save_epoch_step': 2000, 'eval_batch_step': [0, 59], 'cal_metric_during_train': False, 'save_inference_dir': None, 'use_visualdl': False, 'seed': 2022, 'infer_img': '', 'save_res_path': './output/ser/xfund_zh/res', 'kie_rec_model_dir': path1+'\\inferenced_models\\ch_PP-OCRv4_rec_infer', 'kie_det_model_dir': path1+'\\inferenced_models\\det', 'infer_mode': True, 'det_algorithm': 'SAST', 'det_model_dir': path1+'\\inferenced_models\\det', 'rec_model_dir': path1+'\\inferenced_models\\ch_PP-OCRv4_rec_infer', 'cls_model_dir': path1+'\\inferenced_models\\cls', 'kie_cls_model_dir': path1+'\\inferenced_models\\cls', 'class_path': path1+'\\inferenced_models\\\\ser\\\\allLabel.txt', 'distributed': False}, 'Architecture': {'model_type': 'kie', 'algorithm': 'LayoutXLM', 'Transform': None, 'Backbone': {'name': 'LayoutXLMForSer', 'pretrained': '/data/kath/sources/PaddleOCR-release-2.7/output/BigModel/best_accuracy', 'checkpoints': path1+'\\inferenced_models\\ser\\inference', 'mode': 'vi', 'num_classes': 8967}}, 'Loss': {'name': 'VQASerTokenLayoutLMLoss', 'num_classes': 8967, 'key': 'backbone_out'}, 'Optimizer': {'name': 'AdamW', 'beta1': 0.9, 'beta2': 0.999, 'lr': {'name': 'Linear', 'learning_rate': 5e-05, 'epochs': 200, 'warmup_epoch': 1}, 'regularizer': {'name': 'L2', 'factor': 0.0}}, 'PostProcess': {'name': 'VQASerTokenLayoutLMPostProcess', 'class_path': path1+'\\inferenced_models\\ser\\allLabel.txt'}, 'Metric': {'name': 'VQASerTokenMetric', 'main_indicator': 'hmean'}, 'Train': {'dataset': {'name': 'SimpleDataSet', 'data_dir': '/data/kath/data/Datasets/', 'label_file_list': ['/data/kath/data/Datasets/train.txt'], 'ratio_list': [1.0], 'transforms': [{'DecodeImage': {'img_mode': 'RGB', 'channel_first': False}}, {'VQATokenLabelEncode': {'contains_re': False, 'algorithm': 'LayoutXLM', 'class_path': '/data/kath/data/Datasets/allLabel.txt', 'use_textline_bbox_info': True, 'order_method': 'tb-yx'}}, {'VQATokenPad': {'max_seq_len': 512, 'return_attention_mask': True}}, {'VQASerTokenChunk': {'max_seq_len': 512}}, {'Resize': {'size': [224, 224]}}, {'NormalizeImage': {'scale': 1, 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'order': 'hwc'}}, {'ToCHWImage': None}, {'KeepKeys': {'keep_keys': ['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'image', 'labels']}}]}, 'loader': {'shuffle': True, 'drop_last': False, 'batch_size_per_card': 16, 'num_workers': 0}}, 'Eval': {'dataset': {'name': 'SimpleDataSet', 'data_dir': '/data/kath/data/Datasets/', 'label_file_list': ['/data/kath/data/Datasets/val.txt'], 'transforms': [{'DecodeImage': {'img_mode': 'RGB', 'channel_first': False}}, {'VQATokenLabelEncode': {'contains_re': False, 'algorithm': 'LayoutXLM', 'class_path': '/data/kath/data/Datasets/allLabel.txt', 'use_textline_bbox_info': True, 'order_method': 'tb-yx'}}, {'VQATokenPad': {'max_seq_len': 512, 'return_attention_mask': True}}, {'VQASerTokenChunk': {'max_seq_len': 512}}, {'Resize': {'size': [224, 224]}}, {'NormalizeImage': {'scale': 0.8, 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'order': 'hwc'}}, {'ToCHWImage': None}, {'KeepKeys': {'keep_keys': ['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'image', 'labels']}}]}, 'loader': {'shuffle': False, 'drop_last': False, 'batch_size_per_card': 16, 'num_workers': 0}}, 'profiler_options': None}




if __name__ == '__main__':
    path=r"D:\kath-workfile\Data\营业执照\dataset\taxi_invoice\data\taxiinvoice_00000040.jpg"
    img = cv2.imread(path)
    text_pred_res_det(img,path,'aaa')