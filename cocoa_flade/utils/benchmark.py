import time
import numpy as np
from typing import Dict, List
from collections import defaultdict

import torch
from torchvision.ops import box_iou
from contextlib import ContextDecorator


class Timer(ContextDecorator):
    """
    Derive from https://github.com/ultralytics/yolov5/blob/master/utils/general.py
    """
    def __init__(self, device: str = None):
        self.total_num  = 0
        self.total_time = 0
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))
        if self.cuda: 
            import torch

    def __enter__(self):
        """
        Initializes timing at the start of a context block.
        """
        self.start = self._time()
        return self

    def __exit__(self, *exec):
        """
        Updating duration upon exiting a context block.
        """
        self.total_time += self._time() - self.start  # delta-time

    def _time(self):
        """Measures and returns the current time, synchronizing CUDA operations if `cuda` is True."""
        if self.cuda: torch.cuda.synchronize(self.device)
        return time.time()
    
    def count(self, num):
        self.total_num += num
    
    def report(self):
        stats = {
            'num':  self.total_num,
            'time': self.total_time,
            'fps':  self.total_num / self.total_time if self.total_time != 0 else -1
        }

        brief = f"Speed: {stats['fps']:5.3f} fps, {1 / stats['fps'] * 1e3:5.3f} ms"

        return stats, brief


class Metric(object):
    def __init__(self, cats: List = None, tags: List = None):
        # information of datasets
        self.cat_info = {cat['id']: cat for cat in cats}
        self.aet_info = {tag['id']: tag for tag in tags}

        # re-id
        self.cat_ids = {cat['name']: i for i, cat in enumerate(cats)}
        self.aet_ids = {tag['name']: i for i, tag in enumerate(tags)}

        # intrinsic parameters
        self.max_dets = {
            'single': 1,
            'dense':  10,
            'infty':  100,
        }
        self.area_rng = {
            'all':    [0 ** 2, 1e5 ** 2],
            'small':  [0 ** 2, 32 ** 2],
            'medium': [32 ** 2, 96 ** 2],
            'large':  [96 ** 2, 1e5 ** 2]
        }
        self.iou_thr = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.rec_thr = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)

        # statistic members
        self._eval  = None
        self._ious  = None
        self._stats = None
        self._gts = defaultdict(list)
        self._dts = defaultdict(list)

    def update(self, outputs, targets):
        # convert to pycoco type
        for output, target in zip(outputs, targets):
            # mapping to image id
            image_id = self.aet_ids[target['name']]
            
            # obtain basic info
            width, height = target['resolution']

            # convert target
            for tgt_cls, tgt_box in zip(target['labels'], target['bboxes']):
                # index check
                if tgt_cls not in self.cat_info.keys(): continue

                # parse elements
                xtl, ytl, w, h = tgt_box
                category_name  = self.cat_info[tgt_cls]['name']
                category_id    = self.cat_ids[category_name]
                
                # emplace back
                self._gts[image_id, category_id].append({
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox': [xtl, ytl, xtl + w, ytl + h],
                    'area': (w * width) * (h * height),
                    'iscrowd': 0,
                    'ignore': 0
                })

            # convert output
            for out_cls, out_prob, out_box in zip(output['labels'], output['scores'], output['bboxes']):
                # index check
                if out_cls not in self.cat_info.keys(): continue

                # parse elements
                score = out_prob
                xtl, ytl, w, h = out_box
                category_name  = self.cat_info[out_cls]['name']
                category_id    = self.cat_ids[category_name]

                # emplace back
                self._dts[image_id, category_id].append({
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox': [xtl, ytl, xtl + w, ytl + h],
                    'area': (w * width) * (h * height),
                    'score': score
                })

    def _evaluate(self):
        """
        Run per image evaluation on given images and store results (a list of dict)
        """
        def _compute_iou(aet_id, cat_id):
            gt = self._gts[aet_id, cat_id]
            dt = self._dts[aet_id, cat_id]
            
            if len(gt) == 0 or len(dt) == 0:
                return []

            return box_iou(torch.tensor([d['bbox'] for d in dt]),
                           torch.tensor([g['bbox'] for g in gt])).numpy()
        
        # calculate iou
        self._ious = {
            (aet_id, cat_id): _compute_iou(aet_id, cat_id) \
                for aet_id in self.aet_ids.values()
                for cat_id in self.cat_ids.values()
        }

        # evaluate
        maxDet = max([max_det for _, max_det in self.max_dets.items()])
        self._stats = np.asarray([
            self._evaluate_each(aet_id, cat_id, area, maxDet) \
                for cat_id in self.cat_ids.values()
                for area   in self.area_rng.values()
                for aet_id in self.aet_ids.values()
        ]).reshape(len(self.cat_ids), len(self.area_rng), len(self.aet_ids))

    def _evaluate_each(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        gt = self._gts[imgId,catId]
        dt = self._dts[imgId,catId]

        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]

        # sort highest score first
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]

        # load computed ious
        if len(self._ious[imgId,catId]) > 0:
            ious = self._ious[imgId,catId][:, gtind]
        else:
            ious = self._ious[imgId,catId]

        if len(dt) > 0 and len(gt) > 0:
            pass
        
        # initialize
        T = len(self.iou_thr)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T, G))
        dtm  = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))

        if not len(ious) == 0:
            for tind, t in enumerate(self.iou_thr):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind,gind]
                        m   = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gind + 1  # not equal 0
                    gtm[tind,m]     = dind + 1  # not equal 0
        
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a, T, 0)))
        
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def _accumulate(self):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        '''
        assert self._stats is not None, "Please run evaluate() first."

        # initialize        
        T           = len(self.iou_thr)
        R           = len(self.rec_thr)
        K           = len(self.cat_ids)
        A           = len(self.area_rng)
        M           = len(self.max_dets)
        precision   = -np.ones((T, R, K, A, M)) # -1 for the precision of absent categories
        recall      = -np.ones((T, 1, K, A, M))
        scores      = -np.ones((T, R, K, A, M))

        # retrieve E at each category, area range, and max number of detections
        for k, (cat_name, cat_id) in enumerate(self.cat_ids.items()):
            for a, (area_name, area_range) in enumerate(self.area_rng.items()):
                for m, (maxDet_name, max_det) in enumerate(self.max_dets.items()):
                    E = [e for e in self._stats[k, a, :] if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:max_det] for e in E])

                    # obtain sorted scores
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    # extract
                    dtm  = np.concatenate([e['dtMatches'][:,0:max_det] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:max_det]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0: continue
                    
                    # calculate
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)

                    for t, iou_thr in enumerate(self.iou_thr):
                        tp = tp_sum[t]
                        fp = fp_sum[t]
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,0,k,a,m] = rc[-1]
                        else:
                            recall[t,0,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, self.rec_thr, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)

        self._eval = {
            'counts': [T, R, K, A, M],
            'precision': precision,
            'recall': recall,
            'scores': scores,
        }

    def _compute(self, pr : str = 'P',
                 cal_mean : bool = True, 
                 iou_thr  : int = None, 
                 cat_name : str = 'all', 
                 area_rng : str = 'all', 
                 max_dets : str = 'infty'):
        """
        Compute the mean Average Precision and Recall
        """
        # index
        tind = np.arange(len(self.iou_thr)) if iou_thr is None else np.where(iou_thr == self.iou_thr)[0]
        cind = np.arange(len(self.cat_ids)) if cat_name == 'all' else [self.cat_ids[cat_name]]
        aind = [i for i, aRng in enumerate(self.area_rng.keys()) if aRng == area_rng]
        mind = [i for i, mDet in enumerate(self.max_dets.keys()) if mDet == max_dets]

        # select mode
        if pr == 'P':
            s = self._eval['precision']
        elif pr == 'R':
            s = self._eval['recall']
        else:
            raise ValueError("Invalid value for 'pr'. Expected 'P' or 'R'.")

        # slice
        s = s[tind, :, :, :, :]
        s = s[:, :, cind, :, :]
        s = s[:, :, :, aind, :]
        s = s[:, :, :, :, mind]

        # return value
        if cal_mean is not True:
            return s
        else:
            return np.mean(s[s > -1]) if len(s[s > -1]) != 0 else -1

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        # evaluate
        self._evaluate()
        
        # accumulate
        self._accumulate()

        # smooth function
        def smooth(y, f=0.05):
            """
            Applies box filter smoothing to array `y` with fraction `f`, yielding a smoothed array.
            """
            nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
            p = np.ones(nf // 2)  # ones padding
            yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
            return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed

        # create statistics
        stats = {cat: dict() for cat in [k for k in self.cat_ids.keys()] + ['all']}
        brief = ("%11s" * 7) % ("Class", "P", "R", "F1", "AP50", "AP50-95", "AR100")
        for cat_name, cat_info in stats.items():
            # pre process
            p = self._compute('P', cat_name=cat_name, iou_thr=.50, cal_mean=False).squeeze()
            p = p.mean(-1) if p.ndim != 1 else p
            r = self.rec_thr
            f1 = 2 * p * r / (p + r + 1e-16)
            i = smooth(f1, 0.1).argmax()  # max F1 index

            # F1 score
            cat_info['Precision'] = p[i]
            cat_info['Recall'] = r[i]
            cat_info['F1-score'] = f1[i] 

            # average precision
            cat_info['AP50-95'] = self._compute('P', cat_name=cat_name)
            cat_info['AP50']    = self._compute('P', cat_name=cat_name, iou_thr=.50)
            cat_info['AP75']    = self._compute('P', cat_name=cat_name, iou_thr=.75)
            cat_info['APs']     = self._compute('P', cat_name=cat_name, area_rng='small')
            cat_info['APm']     = self._compute('P', cat_name=cat_name, area_rng='medium')
            cat_info['APl']     = self._compute('P', cat_name=cat_name, area_rng='large')

            # average recall
            cat_info['AR1']   = self._compute('R', cat_name=cat_name, max_dets='single')
            cat_info['AR10']  = self._compute('R', cat_name=cat_name, max_dets='dense')
            cat_info['AR100'] = self._compute('R', cat_name=cat_name, max_dets='infty')
            cat_info['ARs']   = self._compute('R', cat_name=cat_name, area_rng='small')
            cat_info['ARm']   = self._compute('R', cat_name=cat_name, area_rng='medium')
            cat_info['ARl']   = self._compute('R', cat_name=cat_name, area_rng='large')

            # append printable brief
            brief += "\n%11s" % cat_name.capitalize()
            brief += ("%11.3f" * 6) % (
                cat_info['Precision'],
                cat_info['Recall'],
                cat_info['F1-score'],
                cat_info['AP50'],
                cat_info['AP50-95'],
                cat_info['AR100']
            )

        return stats, brief


if __name__ == '__main__':
    torch.manual_seed(114)

    targets = tuple([
        {
            'file': 'test01',
            'labels': torch.tensor([0, 0]).cuda(),
            'bboxes': torch.tensor([[0.1, 0.2, 0.4, 0.4],
                                    [0.8, 0.8, 0.1, 0.1]]).cuda(),
            'resolution': (346, 260)
        }
    ])

    outputs = tuple([
        {
            'scores': torch.tensor([0.9, 0.8, 0.8]).cuda(),
            'labels': torch.tensor([0, 0, 0]).cuda(),
            'bboxes': torch.tensor([[0.11, 0.21, 0.39, 0.35],
                                    [0.79, 0.79, 0.10, 0.10],
                                    [0.50, 0.60, 0.10, 0.10]]).cuda(),
        }
    ])

    eval = Metric(aet_ids={'test00': 0, 'test01': 1, 'test02': 2}, 
                  cat_ids={'fire': 0, 'book': 1, 'dog': 2, 'cat': 3, 'fly': 4, 'brush': 5})
    eval.update(outputs, targets)
    eval.summarize()
