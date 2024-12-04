import cv2
import math
import numpy as np
from . import extractor_ops


class FeatExtractor(extractor_ops.Converter):
    def __init__(self, resolution) -> None:
        super().__init__(resolution)
        self.resolution = resolution

    def process(self, sample, bboxes):
        if len(bboxes) == 0:
            return []

        feats = []
        for box in bboxes:  
            feats.append(super().process(sample['events'], box))

        # print(feats)
        # print([self.extract_feats(sample, box) for box in bboxes])

        return feats
        # return [self.extract_feats(sample, box) for box in bboxes]

    def extract_feats(self, sample, box):
        # rescale
        box_x = box[0] * self.resolution[0]
        box_y = box[1] * self.resolution[1]
        box_w = box[2] * self.resolution[0]
        box_h = box[3] * self.resolution[1]

        # filter
        idn = np.logical_and(
            np.logical_and(sample['events'][:, 1] >= box_x, sample['events'][:, 1] <= (box_x + box_w)),
            np.logical_and(sample['events'][:, 2] >= box_y, sample['events'][:, 2] <= (box_y + box_h))
        )
        events = sample['events'][idn]

        # predefine functions
        def project(events):
            image = np.zeros(self.resolution)
            image[events[:, 1], events[:, 2]] = 255
            return image.astype(np.uint8)

        def mcontour(count):
            contours = cv2.findContours(count, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            
            return contours[0]

        def moment(contour):
            M = cv2.moments(contour)
            cX = int(M["m10"] / (M["m00"] + np.finfo(float).eps))
            cY = int(M["m01"] / (M["m00"] + np.finfo(float).eps))
            return cX, cY

        # project
        count = project(events)

        # contours
        contour = mcontour(count)
        area = cv2.contourArea(contour)
        arc_length = cv2.arcLength(contour, True) + 1

        # corners
        # harris = cv2.cornerHarris(count, blockSize=2, ksize=3, k=0.04)
        max_corners = 20
        corners = cv2.goodFeaturesToTrack(count, maxCorners=max_corners, qualityLevel=0.01, minDistance=10)

        # wrap events into buffers
        buffer = np.split(events, [int(len(events) / 2)])
        buffer = [buf for buf in buffer if len(buf) != 0]
        buf_contours = [
            mcontour(project(buffer)) for buffer in buffer
        ]

        # buffer areas
        buf_areas = [
            cv2.contourArea(buf_contour) + 1 for buf_contour in buf_contours 
        ]

        # buffer movements
        buf_moments = np.array([
            moment(buf_countour) for buf_countour in buf_contours
        ])
        buf_movements = np.array([
            buf_moments[i] - buf_moments[i - 1] for i in range(1, len(buf_moments))
        ]) if len(buf_moments) > 1 else np.array([[0, 0]])

        feat = [
            len(events) / (box_w * box_h),  # 事件输出率
            box_w / box_h,          # 长宽比
            area / (box_w * box_h), # 矩形度
            4 * math.pi * area / arc_length ** 2, # 圆形度
            abs(buf_areas[-1] - buf_areas[0]) * len(buf_areas) / sum(buf_areas), # 面积变化率
            np.linalg.norm(buf_movements, axis=1).sum(), # 质心移动
            len(corners) / max_corners, # features 角点
            # (harris != 0).sum(), # harris 角点
        ]

        return feat
