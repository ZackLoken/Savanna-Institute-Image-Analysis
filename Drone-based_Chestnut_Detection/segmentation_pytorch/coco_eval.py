import copy
import io
import json
from contextlib import redirect_stdout
from collections import defaultdict
import numpy as np
import pycocotools.mask as mask_util
import torch
from segmentation_pytorch import utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params

# --- Helper metric functions ---
def compute_binary_iou(pred_mask, gt_mask, smooth=1e-6):
    """
    Compute IoU between two binary masks.
    Both pred_mask and gt_mask should be numpy arrays (dtype=uint8) with values {0,1}.
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return (intersection + smooth) / (union + smooth)

def compute_dice_score(pred_mask, gt_mask, smooth=1e-6):
    """
    Compute Dice score between two binary masks.
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    return (2.0 * intersection + smooth) / (pred_mask.sum() + gt_mask.sum() + smooth)

# --- CustomParams ---
class CustomParams(Params):
    class CustomParams(Params):
        def setDetParams(self):
            self.imgIds = []
            self.catIds = []  # Keep as empty list
            # Use original IoU thresholds
            self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
            self.maxDets = [1]  # One detection per image for semantic segmentation
            # For 224x224 images
            img_area = 224 * 224
            self.areaRng = [
                [0, img_area],             # all
                [0, img_area * 0.05],      # small
                [img_area * 0.05, img_area * 0.15],  # medium
                [img_area * 0.15, img_area]  # large
            ]
            self.areaRngLbl = ['all', 'small', 'medium', 'large']
            self.useCats = 1

        def __init__(self, iouType='segm'):
            if iouType == 'segm' or iouType == 'bbox':
                self.setDetParams()
            else:
                raise Exception('Only segmentation evaluation supported')
            self.iouType = iouType
            self.useSegm = None

# --- CustomCOCOeval ---
class CustomCOCOeval(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        """
        Initialize CustomCOCOeval using COCO APIs for gt and dt.
        """
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt = cocoGt              # ground truth COCO API
        self.cocoDt = cocoDt              # detections COCO API
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI]
        self.eval = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = CustomParams(iouType=iouType) # use CustomParams
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # will store [mean_iou, mean_dice]
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def summarize(self):
        """
        Compute and display binary segmentation performance metrics:
        Mean IoU and Mean Dice score.
        """
        mean_iou, mean_dice = self.summarize_segmentation_metrics()
        print(f"Mean IoU: {mean_iou:.3f}")
        print(f"Mean Dice: {mean_dice:.3f}")
        self.stats = [mean_iou, mean_dice]
        return self.stats

    def summarize_segmentation_metrics(self):
        """
        For each image in the evaluation, decode the predicted segmentation
        and the ground-truth mask, then compute the IoU and Dice score.
        """
        ious = []
        dices = []
        img_ids = self.params.imgIds
        for img_id in img_ids:
            # Retrieve ground truth annotations for this image.
            gt_ann_ids = self.cocoGt.getAnnIds(imgIds=img_id, catIds=self.params.catIds, iscrowd=None)
            gt_anns = self.cocoGt.loadAnns(gt_ann_ids)
            gt_mask = np.zeros((224, 224), dtype=np.uint8)  # assuming images are 224x224 as set in params
            for ann in gt_anns:
                # Decode RLE segmentation and combine using a logical OR.
                rle = ann['segmentation']
                mask = mask_util.decode(rle)
                gt_mask = np.logical_or(gt_mask, mask).astype(np.uint8)
            # Retrieve detection results for this image.
            dt_ann_ids = self.cocoDt.getAnnIds(imgIds=img_id, catIds=self.params.catIds)
            dt_anns = self.cocoDt.loadAnns(dt_ann_ids)
            if len(dt_anns) == 0:
                continue  # No detection for this image.
            # Use the first detection (assuming one per image).
            dt_ann = dt_anns[0]
            dt_rle = dt_ann['segmentation']
            dt_mask = mask_util.decode(dt_rle)
            dt_mask = (dt_mask > 0).astype(np.uint8)
            # Compute metrics.
            iou = compute_binary_iou(dt_mask, gt_mask)
            dice = compute_dice_score(dt_mask, gt_mask)
            ious.append(iou)
            dices.append(dice)
        mean_iou = np.mean(ious) if ious else 0
        mean_dice = np.mean(dices) if dices else 0
        return mean_iou, mean_dice

    def accumulate(self):
        # We leave this method mostly intact if needed by the COCO API, but note we no longer use AP/AR.
        for imgId in self.params.imgIds:
            for catId in self.params.catIds:
                iou = self.computeIoU(imgId, catId)
                self.ious[(imgId, catId)] = iou

    def _prepare(self):
        # Use COCOeval's original _prepare
        super()._prepare()

    def evaluate(self):
        """
        Run per-image evaluation for semantic segmentation. This will still call the COCOeval routines
        to compute overlaps, but ultimately our new summarize() will report IoU and Dice.
        """
        p = self.params
        p.iouType = 'segm'
        p.imgIds = list(np.unique(p.imgIds))
        p.catIds = [1]  # single class for binary segmentation
        p.useCats = True
        p.maxDets = [1]
        self.params = p
        self._prepare()
        self.ious = {}
        for imgId in p.imgIds:
            for catId in p.catIds:
                iou = self.computeIoU(imgId, catId)
                self.ious[(imgId, catId)] = iou
        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        evalImgs = [
            evaluateImg(imgId, catId, areaRng, maxDet)
            for catId in p.catIds
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]
        evalImgs = np.asarray(evalImgs).reshape(len(p.catIds), len(p.areaRng), len(p.imgIds))
        self._paramsEval = copy.deepcopy(self.params)
        return p.imgIds, evalImgs

# --- CustomCocoEvaluator ---
class CustomCocoEvaluator(object):
    """
    COCO evaluator for semantic segmentation with single binary mask output.
    Adapted from torchvision's COCOEvaluator.
    """
    def __init__(self, coco_gt, iou_types):
        if not isinstance(iou_types, (list, tuple)):
            raise TypeError(f"This constructor expects iou_types of type list or tuple, instead got {type(iou_types)}")
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = CustomCOCOeval(coco_gt, iouType=iou_type)
        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)
        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            with redirect_stdout(io.StringIO()):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)
            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        # For each IoU type, print the custom segmentation metrics (mean IoU and Dice).
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"Segmentation metrics for IoU type: {iou_type}")
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        """
        Route predictions to the appropriate preparation function.
        """
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
            try:
                masks = prediction["masks"]  # Expected shape: [B, 1, H, W]
                if len(masks.shape) == 3:
                    masks = masks.unsqueeze(1)
                for idx in range(masks.shape[0]):
                    mask = masks[idx, 0]
                    score = prediction["scores"][idx]
                    binary_mask = (mask > 0.5).cpu().numpy().astype(np.uint8)
                    rle = mask_util.encode(np.asfortranarray(binary_mask))
                    if isinstance(rle, list):
                        rle = rle[0]
                    rle['counts'] = rle['counts'].decode('utf-8')
                    result = {
                        "image_id": int(original_id),
                        "category_id": 1,
                        "segmentation": rle,
                        "score": float(score)
                    }
                    coco_results.append(result)
            except Exception as e:
                print(f"Error processing prediction for image {original_id}: {e}")
                continue
        return coco_results

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def merge(img_ids, eval_imgs):
    all_img_ids = utils.all_gather(img_ids)
    all_eval_imgs = utils.all_gather(eval_imgs)
    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)
    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)
    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]
    return merged_img_ids, merged_eval_imgs

def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())
    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################

# Ideally, pycocotools wouldn't have hard-coded prints
# so that we could avoid copy-pasting those two functions

def createIndex(self):
    # create index
    # print('creating index...')
    anns, cats, imgs = {}, {}, {}
    imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
    if 'annotations' in self.dataset:
        for ann in self.dataset['annotations']:
            imgToAnns[ann['image_id']].append(ann)
            anns[ann['id']] = ann

    if 'images' in self.dataset:
        for img in self.dataset['images']:
            imgs[img['id']] = img

    if 'categories' in self.dataset:
        for cat in self.dataset['categories']:
            cats[cat['id']] = cat

    if 'annotations' in self.dataset and 'categories' in self.dataset:
        for ann in self.dataset['annotations']:
            catToImgs[ann['category_id']].append(ann['image_id'])

    # print('index created!')

    # create class members
    self.anns = anns
    self.imgToAnns = imgToAnns
    self.catToImgs = catToImgs
    self.imgs = imgs
    self.cats = cats


maskUtils = mask_util


def loadRes(self, resFile):
    """
    Load result file and return a result api object.
    Args:
        self (obj): coco object with ground truth annotations
        resFile (str): file name of result file
    Returns:
    res (obj): result api object
    """
    res = COCO()
    res.dataset['images'] = [img for img in self.dataset['images']]

    # print('Loading and preparing results...')
    # tic = time.time()
    if isinstance(resFile, str):
        anns = json.load(open(resFile))
    elif type(resFile) == np.ndarray:
        anns = self.loadNumpyAnnotations(resFile)
    else:
        anns = resFile
    assert type(anns) == list, 'results in not an array of objects'
    annsImgIds = [ann['image_id'] for ann in anns]
    assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
        'Results do not correspond to current coco set'
    if 'caption' in anns[0]:
        imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
        res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
        for id, ann in enumerate(anns):
            ann['id'] = id + 1
    elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for id, ann in enumerate(anns):
            bb = ann['bbox']
            x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
            if 'segmentation' not in ann:
                ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            ann['area'] = bb[2] * bb[3]
            ann['id'] = id + 1
            ann['iscrowd'] = 0
    elif 'segmentation' in anns[0]:
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for id, ann in enumerate(anns):
            # now only support compressed RLE format as segmentation results
            ann['area'] = maskUtils.area(ann['segmentation'])
            if 'bbox' not in ann:
                ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
            ann['id'] = id + 1
            ann['iscrowd'] = 0
    elif 'keypoints' in anns[0]:
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for id, ann in enumerate(anns):
            s = ann['keypoints']
            x = s[0::3]
            y = s[1::3]
            x1, x2, y1, y2 = np.min(x), np.max(x), np.min(y), np.max(y)
            ann['area'] = (x2 - x1) * (y2 - y1)
            ann['id'] = id + 1
            ann['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    # print('DONE (t={:0.2f}s)'.format(time.time()- tic))

    res.dataset['annotations'] = anns
    createIndex(res)
    return res


def evaluate(self):
    '''
    Run per image evaluation on given images and store results for semantic segmentation
    :return: tuple of image IDs and evaluation results
    '''
    p = self.params
    
    # Force segmentation evaluation
    p.iouType = 'segm'
    
    # Ensure unique image IDs
    p.imgIds = list(np.unique(p.imgIds))
    
    # For semantic segmentation, we only have one category
    p.catIds = [1]  # Single class (foreground)
    p.useCats = True
    
    # Single detection per image for semantic segmentation
    p.maxDets = [1]
    
    self.params = p
    self._prepare()

    # Compute IoU
    self.ious = {}
    for imgId in p.imgIds:
        for catId in p.catIds:
            iou = self.computeIoU(imgId, catId)
            self.ious[(imgId, catId)] = iou

    # Evaluate each image
    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in p.catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]

    # Reshape evaluation results
    evalImgs = np.asarray(evalImgs).reshape(len(p.catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)

    return p.imgIds, evalImgs