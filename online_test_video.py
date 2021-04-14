import os
import glob
import json
import pandas as pd
import numpy as np
import csv
import torch
import time
from torch.autograd import Variable
from PIL import Image
import cv2
from torch.nn import functional as F
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    DatasetEvaluators,
    PascalVOCDetectionEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.pascal_voc import register_pascal_voc

_datasets_root = "datasets"
for d in ["trainval", "test"]:
    register_pascal_voc(name=f'100DOH_hand_{d}', dirname=_datasets_root, split=d, year=2007, class_names=["hand"])
    MetadataCatalog.get(f'100DOH_hand_{d}').set(evaluator_type='pascal_voc')

from opts import parse_opts_online
from get_model import generate_model
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel

#from dataset import get_online_data
from utils import  AverageMeter, LevenshteinDistance, Queue

import pdb
import numpy as np
import datetime

from demo.demo_options import DemoOptions
from bodymocap.body_mocap_api import BodyMocap
from handmocap.hand_mocap_api import HandMocap
import mocap_utils.demo_utils as demo_utils
import mocap_utils.general_utils as gnu
from mocap_utils.timer import Timer
from datetime import datetime

from handmocap.hand_bbox_detector import HandBboxDetector
from bodymocap.body_bbox_detector import BodyPoseEstimator

###Pretrained RGB models
##Google Drive
#https://drive.google.com/file/d/1V23zvjAKZr7FUOBLpgPZkpHGv8_D-cOs/view?usp=sharing
##Baidu Netdisk
#https://pan.baidu.com/s/114WKw0lxLfWMZA6SYSSJlw code:p1va

def weighting_func(x):
    return (1 / (1 + np.exp(-0.2 * (x - 9))))

def __filter_bbox_list(body_bbox_list,hand_bbox_detector,single_person):
    bbox_size =  [ (x[2] * x[3]) for x in body_bbox_list]
    idx_big2small = np.argsort(bbox_size)[::-1]
    body_bbox_list = [ body_bbox_list[i] for i in idx_big2small ]
    hand_bbox_list = [hand_bbox_list[i] for i in idx_big2small]

    if single_person and len(body_bbox_list)>0:
        body_bbox_list = [body_bbox_list[0], ]
        hand_bbox_list = [hand_bbox_list[0], ]

    return body_bbox_list, hand_bbox_list

def __pad_and_resize(self, img, hand_bbox, add_margin, final_size=224):
    ori_height, ori_width = img.shape[:2]
    min_x, min_y = hand_bbox[:2].astype(np.int32)
    width, height = hand_bbox[2:].astype(np.int32)
    max_x = min_x + width
    max_y = min_y + height

    if width > height:
        margin = (width-height) // 2
        min_y = max(min_y-margin, 0)
        max_y = min(max_y+margin, ori_height)
    else:
        margin = (height-width) // 2
        min_x = max(min_x-margin, 0)
        max_x = min(max_x+margin, ori_width)
    
    # add additional margin
    if add_margin:
        margin = int(0.3 * (max_y-min_y)) # if use loose crop, change 0.3 to 1.0
        min_y = max(min_y-margin, 0)
        max_y = min(max_y+margin, ori_height)
        min_x = max(min_x-margin, 0)
        max_x = min(max_x+margin, ori_width)

    img_cropped = img[int(min_y):int(max_y), int(min_x):int(max_x), :]
    new_size = max(max_x-min_x, max_y-min_y)
    new_img = np.zeros((new_size, new_size, 3), dtype=np.uint8)
    # new_img = np.zeros((new_size, new_size, 3))
    new_img[:(max_y-min_y), :(max_x-min_x), :] = img_cropped
    bbox_processed = (min_x, min_y, max_x, max_y)

    # resize to 224 * 224
    new_img = cv2.resize(new_img, (final_size, final_size))

    ratio = final_size / new_size
    return new_img, ratio, (min_x, min_y, max_x-min_x, max_y-min_y)

def __process_hand_bbox(raw_image, hand_bbox, hand_type, add_margin=True):
        assert hand_type in ['left_hand', 'right_hand']
        img_cropped, bbox_scale_ratio, bbox_processed = __pad_and_resize(raw_image, hand_bbox, add_margin)
    
        if hand_type=='left_hand':
            img_cropped = np.ascontiguousarray(img_cropped[:, ::-1,:], img_cropped.dtype) 
        else:
            assert hand_type == 'right_hand'

        return img_cropped, bbox_scale_ratio, bbox_processed

def get_bbox(bbox_detector,frame):
    detect_output = bbox_detector.detect_hand_bbox(frame) ###(cv2)
    body_pose_list, body_bbox_list, hand_bbox_list,raw_hand_bboxes = detect_output

    if len(hand_bbox_list)<1:
        print('No hand detected')

    cond1 = len(body_bbox_list) > 0 or len(hand_bbox_list)>0
    cond2 = len(body_bbox_list) > 0 and len(hand_bbox_list) == 0
    if cond1:
        body_pose_list,body_bbox_list,hand_bbox_list = __filter_bbox_list(body_bbox_list,hand_bbox_list,single_person)
        if cond2:
            _,body_bbox_list = bbox_detector.detect_body_bbox(frame)
            if len(body_bbox_list) < 1:
                return list()
            hand_bbox_list = [None, ] * len(body_bbox_list)
            body_bbox_list, _ = __filter_bbox_list(body_bbox_list, hand_bbox_list, args.single_person)
            pred_body_list = body_mocap.regress(frame,body_bbox_list)
            hand_bbox_list = body_mocap.get_hand_bboxes(pred_body_list,frame.shape[:2])         
    return hand_bbox_list

def crop_frame(hand_bbox_list,frame):
    for hand_bboxes in hand_bbox_list:
        if hand_bboxes is None:
            return None
    for hand_type in hand_bboxes:
        bbox = hand_bboxes[hand_type]
        if bbox is None:
            return None
        else: 
            img_cropped, bbox_scale_ratio, bbox_processed = __process_hand_bbox(frame,hand_bboxes[hand_type],hand_type,add_margin = False)
            return img_cropped

def detected_hand(frame):
    hand_bbox_list = get_bbox(bbox_detector, frame)
    cropped_frame = crop_frame(hand_bbox_list,frame)
    return cropped_frame

opt = parse_opts_online()

def pp_checkpoint(checkpoint):
    
    new_state_dict = OrderedDict()
    
    for k, v in checkpoint.items():
        name = k[7:]
        new_state_dict[name] = v
    
    return new_state_dict

def load_models(opt):
    opt.resume_path = opt.resume_path_det
    opt.pretrain_path = "results/egogesture_resnetl_10_RGB_8.pth"
    opt.sample_duration = 8
    opt.model = 'resnetl'
    opt.model_clf = 'resnext'
    opt.model_det = 'resnetl'
    opt.model_depth = 10
    opt.width_mult = 0.5
    opt.modality = 'RGB'
    opt.resnet_shortcut = 'A'
    opt.n_classes = 2
    opt.n_finetune_classes = 2

    opt.width_mult_det = 0.5
    opt.width_mult_clf = 1
    opt.model_depth_det = 10
    opt.model_depth_clf = 101
    opt.n_threads = 16
    opt.modality_det = 'RGB'
    opt.modality_clf = 'RGB'
    opt.det_queue_size = 1

    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)

    print(opt)
    with open(os.path.join(opt.result_path, 'opts_det.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    detector, parameters = generate_model(opt)
    detector = detector.cuda()
    if opt.resume_path:
        opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        print('loading checkpoint {}'.format(opt.resume_path))
        #pp_checkpoint(checkpoint)
        checkpoint = torch.load("results/egogesture_resnetl_10_RGB_8.pth")
        newcheckpoint = pp_checkpoint(checkpoint['state_dict'])
        detector.load_state_dict(newcheckpoint)

    print('Model 1 \n', detector)
    pytorch_total_params = sum(p.numel() for p in detector.parameters() if
                               p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    opt.resume_path = opt.resume_path_clf
    opt.pretrain_path = "results/egogesture_resnext_101_RGB_32.pth"
    opt.sample_duration = 32
    opt.model = "resnext"
    opt.model_depth = 101
    opt.width_mult = 1
    opt.modality = "RGB"
    opt.resnet_shortcut = "B"
    opt.n_classes = 83
    opt.n_finetune_classes = 83
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)

    print(opt)
    with open(os.path.join(opt.result_path, 'opts_clf.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)
    classifier, parameters = generate_model(opt)
    classifier = classifier.cuda()
    #if opt.resume_path:
    print('loading checkpoint {}'.format(opt.resume_path))
    checkpoint = torch.load('results/egogesture_resnext_101_RGB_32.pth')
    newcheckpoint = pp_checkpoint(checkpoint['state_dict'])
    classifier.load_state_dict(newcheckpoint)

    print('Model 2 \n', classifier)
    pytorch_total_params = sum(p.numel() for p in classifier.parameters() if
                               p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    return detector, classifier

def pad_and_resize(img, hand_bbox, add_margin,fwidth,fheight):
    
    ori_height, ori_width = img.shape[:2]
    min_x, min_y = hand_bbox[:2].astype(np.int32)
    width, height = hand_bbox[2:].astype(np.int32)
    max_x = min_x + width
    max_y = min_y + height

    if width > height:
        margin = (width-height) // 2
        min_y = max(min_y-margin, 0)
        max_y = min(max_y+margin, ori_height)
    else:
        margin = (height-width) // 2
        min_x = max(min_x-margin, 0)
        max_x = min(max_x+margin, ori_width)
    
    # add additional margin
    if add_margin:
        margin = int(0.3 * (max_y-min_y)) # if use loose crop, change 0.3 to 1.0
        min_y = max(min_y-margin, 0)
        max_y = min(max_y+margin, ori_height)
        min_x = max(min_x-margin, 0)
        max_x = min(max_x+margin, ori_width)

    img_cropped = img[int(min_y):int(max_y), int(min_x):int(max_x), :]
    new_size = max(max_x-min_x, max_y-min_y)
    new_img = np.zeros((new_size, new_size, 3), dtype=np.uint8)
    # new_img = np.zeros((new_size, new_size, 3))
    new_img[:(max_y-min_y), :(max_x-min_x), :] = img_cropped
    bbox_processed = (min_x, min_y, max_x, max_y)

    # resize to 224 * 224
    new_img = cv2.resize(new_img, (fwidth,fheight))

    ratio = 224 / new_size
    return new_img, ratio, (min_x, min_y, max_x-min_x, max_y-min_y)
# /home/jumi/Real-time-GesRec/extra_data/hand_module/hand_detector/model_0529999.pth
class hand_detector():
    def __init__(self):
        self.cfg = get_cfg()
       
        self.cfg.merge_from_file("detectors/hand_only_detector/faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml")
        # cfg.MODEL.WEIGHTS = 'models/model_0529999.pth' # add model weight here
        self.cfg.MODEL.WEIGHTS = 'extra_data/hand_module/hand_detector/model_0529999.pth'
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.predictor = DefaultPredictor(self.cfg)
        '''
        self._datasets_root = 'datasets'
        for d in ["trainval", "test"]:
            register_pascal_voc(name=f'100DOH_hand_{d}', dirname=self._datasets_root, split=d, year=2007, class_names=["hand"])
            MetadataCatalog.get(f'100DOH_hand_{d}').set(evaluator_type='pascal_voc')
        '''

    def detect_hand(self,frame):
        outputs = self.predictor(frame)
        
        pred_boxes = outputs['instances'].pred_boxes.tensor.cpu().numpy()
        if len(pred_boxes) == 0:
            return frame,frame
        pred_boxes = pred_boxes[0]
        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        infer_img = v.get_image()[:, :, ::-1]
        x0,y0,x1,y1 = pred_boxes
        w = x1 - x0
        h = y1 - y0
        pred_boxes = np.array([x0,y0,w,h])
        cropped_img,_,_ = pad_and_resize(frame,pred_boxes,True,fwidth=320,fheight=240)
        return cropped_img,infer_img

bboxdetector = hand_detector()

detector, classifier = load_models(opt)

if opt.no_mean_norm and not opt.std_norm:
    norm_method = Normalize([0, 0, 0], [1, 1, 1])
elif not opt.std_norm:
    norm_method = Normalize(opt.mean, [1, 1, 1])
else:
    norm_method = Normalize(opt.mean, opt.std)

spatial_transform = Compose([
    Scale(112),
    CenterCrop(112),
    ToTensor(opt.norm_value), norm_method
])

opt.sample_duration_clf = 32
opt.sample_duration_det = 8

opt.sample_duration = max(opt.sample_duration_clf, opt.sample_duration_det)
fps = ""
cap = cv2.VideoCapture(0)
num_frame = 0
single_person = True
clip = []
sub_clip = []
active_index = 0
passive_count = 0
active = False
prev_active = False
finished_prediction = None
pre_predict = False

device = torch.device('cuda') 
#hand_bbox_detector = HandBboxDetector('third_view',device)

detector.eval()
classifier.eval()
cum_sum = np.zeros(opt.n_classes_clf, )
clf_selected_queue = np.zeros(opt.n_classes_clf, )
det_selected_queue = np.zeros(opt.n_classes_det, )
opt.det_queue_size = 4
opt.clf_queue_size = 32

myqueue_det = Queue(opt.det_queue_size, n_classes=2)
myqueue_clf = Queue(opt.clf_queue_size, n_classes=83)
results = []
prev_best1 = opt.n_classes_clf
spatial_transform.randomize_parameters()
while cap.isOpened():
    t1 = time.time()
    ret, frame = cap.read()
    
    if num_frame == 0:
        cropped_frame,infer_img = bboxdetector.detect_hand(frame)
        cropped_frame = cv2.resize(cropped_frame,(320,240))
        cropped_frame = Image.fromarray(cv2.cvtColor(cropped_frame,cv2.COLOR_BGR2RGB))
        cropped_frame = cropped_frame.convert('RGB')
        # cur_frame = cropped_frame
        cur_frame = cv2.resize(frame,(320,240))
        cur_frame = Image.fromarray(cv2.cvtColor(cur_frame,cv2.COLOR_BGR2RGB))
        cur_frame = cur_frame.convert('RGB')
        for i in range(opt.sample_duration):
            clip.append(cur_frame)
            sub_clip.append(cropped_frame)
        clip = [spatial_transform(img) for img in clip]
        sub_clip = [spatial_transform(img) for img in sub_clip]
    sub_clip.pop(0)
    clip.pop(0)

    sub_frame ,infer_img = bboxdetector.detect_hand(frame)
    sub_frame = cv2.resize(sub_frame,(320,240))
    sub_frame = Image.fromarray(cv2.cvtColor(sub_frame,cv2.COLOR_BGR2RGB))
    sub_frame = sub_frame.convert('RGB')
    sub_frame = spatial_transform(sub_frame)
    sub_clip.append(sub_frame)

    im_dim = clip[0].size()[-2:]
    _frame = cv2.resize(frame,(320,240))
    _frame = Image.fromarray(cv2.cvtColor(_frame,cv2.COLOR_BGR2RGB))
    _frame = _frame.convert('RGB')
    _frame = spatial_transform(_frame)
    clip.append(_frame)
    im_dim = clip[0].size()[-2:]
    try:
        test_data = torch.cat(clip, 0).view((opt.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)
        sub_test_data = torch.cat(sub_clip, 0).view((opt.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)
    except Exception as e:
        pdb.set_trace()
        raise e
    inputs = torch.cat([test_data],0).view(1,3,opt.sample_duration,112,112)
    sub_inputs = torch.cat([sub_test_data],0).view(1,3,opt.sample_duration,112,112)
    num_frame += 1


    ground_truth_array = np.zeros(opt.n_classes_clf + 1, )
    with torch.no_grad():
        #print(inputs.shape)#(1,3,32,112,112)
        inputs = Variable(inputs)
        inputs_det = inputs[:, :, -opt.sample_duration_det:, :, :]
        inputs_det = inputs_det.cuda()
        outputs_det = detector(inputs_det)
        outputs_det = F.softmax(outputs_det, dim=1)
        outputs_det = outputs_det.cpu().numpy()[0].reshape(-1, )
        # enqueue the probabilities to the detector queue
        myqueue_det.enqueue(outputs_det.tolist())

        if opt.det_strategy == 'raw':
            det_selected_queue = outputs_det
        elif opt.det_strategy == 'median':
            det_selected_queue = myqueue_det.median
        elif opt.det_strategy == 'ma':
            det_selected_queue = myqueue_det.ma
        elif opt.det_strategy == 'ewma':
            det_selected_queue = myqueue_det.ewma
        prediction_det = np.argmax(det_selected_queue)

        prob_det = det_selected_queue[prediction_det]
        
        #### State of the detector is checked here as detector act as a switch for the classifier
        if prediction_det == 1:
            inputs_clf = sub_inputs[:, :, :, :, :]
            inputs_clf = torch.Tensor(inputs_clf.numpy()[:,:,::1,:,:])
            inputs_clf = inputs_clf.cuda()
            outputs_clf = classifier(inputs_clf)
            outputs_clf = F.softmax(outputs_clf, dim=1)
            outputs_clf = outputs_clf.cpu().numpy()[0].reshape(-1, )
            # Push the probabilities to queue
            #myqueue_clf.enqueue(outputs_clf.tolist())
            myqueue_clf.enqueue(list(outputs_clf))
            passive_count = 0
            '''
            if opt.clf_strategy == 'raw':
                clf_selected_queue = outputs_clf
            elif opt.clf_strategy == 'median':
                clf_selected_queue = myqueue_clf.median
            elif opt.clf_strategy == 'ma':
                clf_selected_queue = myqueue_clf.ma
            elif opt.clf_strategy == 'ewma':
                clf_selected_queue = myqueue_clf.ewma
            '''
            clf_selected_queue = myqueue_clf.median

        else:
            outputs_clf = np.zeros(opt.n_classes_clf, )
            # Push the probabilities to queue
            #myqueue_clf.enqueue(outputs_clf.tolist())
            myqueue_clf.enqueue(list(outputs_clf))
            passive_count += 1
    
    if passive_count >= opt.det_counter:
        active = False
    else:
        active = True
    
    opt.clf_threshold_pre = 0.3
    opt.clf_threshold_final = 0.15
    # one of the following line need to be commented !!!!
    

    if active:
        active_index += 1
        cum_sum = ((cum_sum * (active_index - 1)) + (weighting_func(active_index) * clf_selected_queue)) / active_index  # Weighted Aproach
        #cum_sum = ((cum_sum * (active_index-1)) + (1.0 * clf_selected_queue))/active_index #Not Weighting Aproach
        best2, best1 = tuple(cum_sum.argsort()[-2:][::1])
        if float(cum_sum[best1] - cum_sum[best2]) > opt.clf_threshold_pre:
            finished_prediction = True
            pre_predict = True

    else:
        active_index = 0
    if active == False and prev_active == True:
        finished_prediction = True
    elif active == True and prev_active == False:
        finished_prediction = False

    opt.stride_len = 1

    if finished_prediction == True:
        #print(finished_prediction,pre_predict)
        best2, best1 = tuple(cum_sum.argsort()[-2:][::1])
        if cum_sum[best1] > opt.clf_threshold_final:
            if pre_predict == True:
                if best1 != prev_best1:
                    if cum_sum[best1] > opt.clf_threshold_final:
                        results.append(((i * opt.stride_len) + opt.sample_duration_clf, best1))
                        print('Early Detected - class : {} with prob : {} at frame {}'.format(best1, cum_sum[best1],
                                                                                              (
                                                                                                          i * opt.stride_len) + opt.sample_duration_clf))
            else:
                if cum_sum[best1] > opt.clf_threshold_final:
                    if best1 == prev_best1:
                        if cum_sum[best1] > 5:
                            results.append(((i * opt.stride_len) + opt.sample_duration_clf, best1))
                            print('Late Detected - class : {} with prob : {} at frame {}'.format(best1,
                                                                                                 cum_sum[best1], (
                                                                                                             i * opt.stride_len) + opt.sample_duration_clf))
                    else:
                        results.append(((i * opt.stride_len) + opt.sample_duration_clf, best1))

                        print('Late Detected - class : {} with prob : {} at frame {}'.format(best1, cum_sum[best1],
                                                                                             (
                                                                                                         i * opt.stride_len) + opt.sample_duration_clf))

            finished_prediction = False
            prev_best1 = best1

        cum_sum = np.zeros(opt.n_classes_clf, )
    
    if active == False and prev_active == True:
        pre_predict = False

    prev_active = active
    elapsedTime = time.time() - t1
    fps = "(Playback) {:.1f} FPS".format(1/elapsedTime)

    if len(results) != 0:
        predicted = np.array(results)[:, 1]
        prev_best1 = -1
    else:
        predicted = []

    print('predicted classes: \t', predicted)

    cv2.putText(frame, fps, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("Result", frame)

    cv2.imshow('Infer_img',infer_img)

    if cv2.waitKey(1)&0xFF == ord('q'):
        break
cv2.destroyAllWindows()

