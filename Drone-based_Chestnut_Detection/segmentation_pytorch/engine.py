import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import segmentation_pytorch.utils
from segmentation_pytorch.coco_eval import CocoEvaluator
from segmentation_pytorch.coco_utils import get_coco_api_from_dataset

def train_one_epoch(model, optimizer, train_data_loader, val_data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    train_metric_logger = segmentation_pytorch.utils.MetricLogger(delimiter="  ")
    train_metric_logger.add_meter('lr', segmentation_pytorch.utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    val_metric_logger = segmentation_pytorch.utils.MetricLogger(delimiter="  ")
    val_metric_logger.add_meter('lr', segmentation_pytorch.utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    train_header = 'Epoch: [{}] Training'.format(epoch)
    val_header = 'Epoch: [{}] Validation'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(train_data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in train_metric_logger.log_every(train_data_loader, print_freq, train_header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            train_loss_dict = model(images, targets)
            train_losses = sum(loss for loss in train_loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        train_loss_dict_reduced = segmentation_pytorch.utils.reduce_dict(train_loss_dict)
        train_losses_reduced = sum(loss for loss in train_loss_dict_reduced.values())

        train_loss_value = train_losses_reduced.item()

        if not math.isfinite(train_loss_value):
            print("Loss is {}, stopping training".format(train_loss_value))
            print(train_loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(train_losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            train_losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        train_metric_logger.update(loss=train_losses_reduced, **train_loss_dict_reduced)
        train_metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # calculate validation loss
    print()
    print("Calculating validation loss...")
    print()

    with torch.no_grad():
        for images, targets in val_metric_logger.log_every(val_data_loader, len(val_data_loader)/2, val_header):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                val_loss_dict = model(images, targets)

            # reduce losses over all GPUs for logging purposes
            val_loss_dict_reduced = segmentation_pytorch.utils.reduce_dict(val_loss_dict)
            val_losses_reduced = sum(loss for loss in val_loss_dict_reduced.values())

            val_loss_value = val_losses_reduced.item()

            if not math.isfinite(val_loss_value):
                print("Loss is {}, stopping training".format(val_loss_value))
                print(val_loss_dict_reduced)
                sys.exit(1)

            val_metric_logger.update(loss=val_losses_reduced, **val_loss_dict_reduced)
            val_metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return train_metric_logger, val_metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, val_data_loader, device, train_data_loader=None):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = segmentation_pytorch.utils.MetricLogger(delimiter="  ")

    val_coco = get_coco_api_from_dataset(val_data_loader.dataset)
    train_coco = get_coco_api_from_dataset(train_data_loader.dataset)
    iou_types = _get_iou_types(model)
    train_coco_evaluator = CocoEvaluator(train_coco, iou_types)
    val_coco_evaluator = CocoEvaluator(val_coco, iou_types)

    # calculate training accuracy
    for images, targets in metric_logger.log_every(train_data_loader, len(train_data_loader) // 3, 'Training:'):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        train_coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    train_coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    train_coco_evaluator.accumulate()
    train_coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    for images, targets in metric_logger.log_every(val_data_loader, len(val_data_loader) // 2, 'Validation:'):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        val_coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    val_coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    val_coco_evaluator.accumulate()
    val_coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    return train_coco_evaluator, val_coco_evaluator


# def train_one_epoch(model, optimizer, train_data_loader, val_data_loader, device, epoch, print_freq, scaler=None):
#     model.train()
#     train_metric_logger = segmentation_pytorch.utils.MetricLogger(delimiter="  ")
#     train_metric_logger.add_meter('lr', segmentation_pytorch.utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     val_metric_logger = segmentation_pytorch.utils.MetricLogger(delimiter="  ")
#     val_metric_logger.add_meter('lr', segmentation_pytorch.utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)

#     lr_scheduler = None
#     if epoch == 0:
#         warmup_factor = 1.0 / 1000
#         warmup_iters = min(1000, len(train_data_loader) - 1)

#         lr_scheduler = torch.optim.lr_scheduler.LinearLR(
#             optimizer, start_factor=warmup_factor, total_iters=warmup_iters
#         )

#     for images, targets in train_metric_logger.log_every(train_data_loader, print_freq, header):
#         images = list(image.to(device) for image in images)
#         targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

#         with torch.amp.autocast('cuda', enabled=scaler is not None):
#             train_loss_dict = model(images, targets)
#             train_losses = sum(loss for loss in train_loss_dict.values())

#         # reduce losses over all GPUs for logging purposes
#         train_loss_dict_reduced = segmentation_pytorch.utils.reduce_dict(train_loss_dict)
#         train_losses_reduced = sum(loss for loss in train_loss_dict_reduced.values())

#         train_loss_value = train_losses_reduced.item()

#         if not math.isfinite(train_loss_value):
#             print("Loss is {}, stopping training".format(train_loss_value))
#             print(train_loss_dict_reduced)
#             sys.exit(1)

#         optimizer.zero_grad()
#         if scaler is not None:
#             scaler.scale(train_losses).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             train_losses.backward()
#             optimizer.step()

#         if lr_scheduler is not None:
#             lr_scheduler.step()

#         train_metric_logger.update(loss=train_losses_reduced, **train_loss_dict_reduced)
#         train_metric_logger.update(lr=optimizer.param_groups[0]["lr"])

#     # calculate validation loss
#     print()
#     print("Calculating validation loss...")
#     print()

#     for images, targets in val_metric_logger.log_every(val_data_loader, len(val_data_loader)/2, header):
#         images = list(image.to(device) for image in images)
#         targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

#         with torch.no_grad():
#             with torch.amp.autocast('cuda', enabled=scaler is not None):
#                 val_loss_dict = model(images, targets)

#             # reduce losses over all GPUs for logging purposes
#             val_loss_dict_reduced = segmentation_pytorch.utils.reduce_dict(val_loss_dict)
#             val_losses_reduced = sum(loss for loss in val_loss_dict_reduced.values())

#             val_loss_value = val_losses_reduced.item()

#             if not math.isfinite(val_loss_value):
#                 print("Loss is {}, stopping training".format(val_loss_value))
#                 print(val_loss_dict_reduced)
#                 sys.exit(1)

#             val_metric_logger.update(loss=val_losses_reduced, **val_loss_dict_reduced)
#             val_metric_logger.update(lr=optimizer.param_groups[0]["lr"])

#     return train_metric_logger, val_metric_logger


# def _get_iou_types(model):
#     model_without_ddp = model
#     if isinstance(model, torch.nn.parallel.DistributedDataParallel):
#         model_without_ddp = model.module
#     iou_types = ["bbox"]
#     if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
#         iou_types.append("segm")
#     if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
#         iou_types.append("keypoints")
#     return iou_types


# @torch.inference_mode()
# def evaluate(model, val_data_loader, device, train_data_loader=None):
#     n_threads = torch.get_num_threads()
#     torch.set_num_threads(1)
#     cpu_device = torch.device("cpu")
#     model.eval()
#     train_metric_logger = segmentation_pytorch.utils.MetricLogger(delimiter="  ")
#     train_header = 'Training Performance:'
#     val_metric_logger = segmentation_pytorch.utils.MetricLogger(delimiter="  ")
#     val_header = 'Validation Performance:'

#     val_coco = get_coco_api_from_dataset(val_data_loader.dataset)
#     train_coco = get_coco_api_from_dataset(train_data_loader.dataset)
#     iou_types = _get_iou_types(model)
#     train_coco_evaluator = CocoEvaluator(train_coco, iou_types)
#     val_coco_evaluator = CocoEvaluator(val_coco, iou_types)

#       # calculate training accuracy
#     for images, targets in train_metric_logger.log_every(train_data_loader, len(train_data_loader)/3, train_header):
#         images = list(img.to(device) for img in images)

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#         model_time = time.time()
#         outputs = model(images)

#         outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
#         model_time = time.time() - model_time

#         res = {target["image_id"]: output for target, output in zip(targets, outputs)}
#         evaluator_time = time.time()
#         train_coco_evaluator.update(res)
#         evaluator_time = time.time() - evaluator_time
#         train_metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

#     # gather the stats from all processes
#     train_metric_logger.synchronize_between_processes()
#     print("Averaged stats:", train_metric_logger)
#     train_coco_evaluator.synchronize_between_processes()

#     # accumulate predictions from all images
#     train_coco_evaluator.accumulate()
#     train_coco_evaluator.summarize()
#     torch.set_num_threads(n_threads)

#     for images, targets in val_metric_logger.log_every(val_data_loader, len(val_data_loader)/2, val_header):
#         images = list(img.to(device) for img in images)

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#         model_time = time.time()
#         outputs = model(images)

#         outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
#         model_time = time.time() - model_time

#         res = {target["image_id"]: output for target, output in zip(targets, outputs)}
#         evaluator_time = time.time()
#         val_coco_evaluator.update(res)
#         evaluator_time = time.time() - evaluator_time
#         val_metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

#     # gather the stats from all processes
#     val_metric_logger.synchronize_between_processes()
#     print("Averaged stats:", val_metric_logger)
#     val_coco_evaluator.synchronize_between_processes()

#     # accumulate predictions from all images
#     val_coco_evaluator.accumulate()
#     val_coco_evaluator.summarize()
#     torch.set_num_threads(n_threads)

#     return train_coco_evaluator, val_coco_evaluator