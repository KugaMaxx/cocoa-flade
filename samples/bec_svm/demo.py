from pathlib import Path
from dataset import FlaDE
from sklearn.svm import SVC
from collections import defaultdict
from cocoa_flade.utils.benchmark import Timer, Metric
from cocoa_flade.utils.draw import plot_projected_events, plot_detection_result, plot_rescaled_image

from generator import BinaryEventContour
from extractor import FeatExtractor

import cv2
import torch
from torchvision.ops import box_iou
from torchvision.ops.boxes import _box_xywh_to_xyxy
import numpy as np


root_path = Path(__file__).resolve().parent
data_path = root_path / '..' / '..' / 'data'

# training
print("Training...")
X_train, Y_train = [], []
data_loader = FlaDE(data_path, 'train', shuffle=True, num_samples=600, denoised=True)
for i, (sample, target) in enumerate(data_loader):
    # create output
    output = dict()
    
    # rois generation, restrict single roi because of unlabeled flying flame
    gen = BinaryEventContour(target['resolution'], candidate_num=1)
    output['bboxes'] = gen.detect(sample)

    # feat extraction
    ext = FeatExtractor(target['resolution'])
    output['feats'] = ext.process(sample, output['bboxes'])

    # whether roi is empty
    if len(output['bboxes']) == 0: continue

    # calculate ious
    output['ious'] = box_iou(_box_xywh_to_xyxy(torch.tensor(output['bboxes'])), 
                             _box_xywh_to_xyxy(torch.tensor(target['bboxes'])))
    
    # construct training input
    for feat, iou in zip(output['feats'], output['ious']):
        # search most candidate label
        ind = iou.argmax()
        if iou[ind] == 0:
            X_train.append(feat)
            Y_train.append(0)
        else:
            X_train.append(feat)
            Y_train.append(target['labels'][ind])

svm = SVC(C=1.0, kernel='rbf', gamma='auto', probability=True)
svm.fit(X_train, Y_train)

# validation
print("Validation...")
data_loader = FlaDE(data_path, 'val', shuffle=True, denoised=True)
timer  = Timer()
metric = Metric(cats=data_loader.dataset.get_cats(), 
                tags=data_loader.dataset.get_tags())
for i, (sample, target) in enumerate(data_loader):
    # ceate output
    output = dict()

    with timer:
        # count number
        timer.count(1)

        # roi generation, restrict single roi because of unlabeled flying flame
        gen = BinaryEventContour(target['resolution'], candidate_num=1)
        output['bboxes'] = gen.detect(sample)

        # feat extraction
        ext = FeatExtractor(target['resolution'])
        output['feats'] = ext.process(sample, output['bboxes'])
        
    # whether roi is empty
    if len(output['bboxes']) == 0:
        output['labels'] = []
        output['scores'] = []
    else:
        output['labels'] = [c for c in svm.predict(output['feats'])]
        output['scores'] = [p.max() for p in svm.predict_proba(output['feats'])]

    metric.update([output], [target])

t_stats, t_brief = timer.report()
print(t_brief)

m_stats, m_brief = metric.summarize()
print(m_brief)
