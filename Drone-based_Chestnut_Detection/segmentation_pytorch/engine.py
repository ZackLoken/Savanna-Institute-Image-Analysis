import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import segmentation_pytorch.utils
from segmentation_pytorch.coco_eval import CocoEvaluator
from segmentation_pytorch.coco_utils import get_coco_api_from_dataset

def train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq, accumulation_steps, val_data_loader=None):
    """
    Train model for one epoch on training dataset and retur training loss to logger. If validation dataset is provided,
    also return validation loss to logger.
    
    :param model: torch.nn.Module
    :param optimizer: torch.optim.Optimizer
    :param train_data_loader: torch.utils.data.DataLoader
    :param device: torch.device
    :param epoch: int
    :param print_freq: int
    :param val_data_loader: torch.utils.data.DataLoader (default=None)
    :param scaler: torch.cuda.amp.GradScaler (default=None)
    :return: train_metric_logger: segmentation_pytorch.utils.MetricLogger, 
             val_metric_logger: segmentation_pytorch.utils.MetricLogger (if val_data_loader is not None)
    """

    model.train()
    train_metric_logger = segmentation_pytorch.utils.MetricLogger(delimiter="  ")
    train_metric_logger.add_meter('lr', segmentation_pytorch.utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    train_header = 'Epoch: [{}] Training'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(train_data_loader) // accumulation_steps - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)

    # Initialize gradient scaler for mixed precision training
    scaler = torch.GradScaler(device)

    for i, (images, targets) in enumerate(train_metric_logger.log_every(train_data_loader, print_freq, train_header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        # Zero the gradients before the forward pass
        if (i % accumulation_steps == 0):
            optimizer.zero_grad()

        with torch.autocast(device_type=device):
            train_loss_dict = model(images, targets)
            train_losses = sum(loss for loss in train_loss_dict.values()) / accumulation_steps

        # reduce losses over all GPUs for logging purposes
        train_loss_dict_reduced = segmentation_pytorch.utils.reduce_dict(train_loss_dict)
        train_losses_reduced = sum(loss for loss in train_loss_dict_reduced.values())
        train_loss_value = train_losses_reduced.item()

        if not math.isfinite(train_loss_value):
            print("Loss is {}, stopping training".format(train_loss_value))
            print(train_loss_dict_reduced)
            sys.exit(1)

        if scaler is not None:
            scaler.scale(train_losses).backward()
        else:
            train_losses.backward()

        if ((i + 1) % accumulation_steps == 0) or (i + 1 == len(train_data_loader)):
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

        train_metric_logger.update(loss=train_losses_reduced, **train_loss_dict_reduced)
        train_metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # validation loop
    if val_data_loader is not None:
        val_metric_logger = segmentation_pytorch.utils.MetricLogger(delimiter="  ")
        val_metric_logger.add_meter('lr', segmentation_pytorch.utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        val_header = 'Epoch: [{}] Validation'.format(epoch)

        with torch.no_grad():
            for images, targets in val_metric_logger.log_every(val_data_loader, len(val_data_loader)/2, val_header):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            
                with torch.autocast(device_type=device):
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

    if val_data_loader is not None:
        return train_metric_logger, val_metric_logger
    else:
        return train_metric_logger


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
def evaluate(model, val_data_loader, val_coco_ds, device, train_data_loader=None, train_coco_ds=None):
    """
    Evaluate model on validation dataset and return validation accuracy to logger. If training dataset is provided,
    also return training accuracy to logger.

    :param model: torch.nn.Module
    :param val_data_loader: torch.utils.data.DataLoader
    :param device: torch.device
    :param train_data_loader: torch.utils.data.DataLoader (default=None)
    :return: train_coco_evaluator: segmentation_pytorch.coco_eval.CocoEvaluator, 
             val_coco_evaluator: segmentation_pytorch.coco_eval.CocoEvaluator (if train_data_loader is not None)
    """

    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    val_metric_logger = segmentation_pytorch.utils.MetricLogger(delimiter="  ")
    train_metric_logger = segmentation_pytorch.utils.MetricLogger(delimiter="  ") if train_data_loader is not None else None

    iou_types = _get_iou_types(model)

    train_coco_evaluator = None
    if train_data_loader is not None:
        if train_coco_ds is None:
            train_coco_ds = get_coco_api_from_dataset(train_data_loader.dataset)
        train_coco_evaluator = CocoEvaluator(train_coco_ds, iou_types)

        # calculate training accuracy
        for images, targets in train_metric_logger.log_every(train_data_loader, len(train_data_loader) // 3, 'Training Accuracy: '):
            images = list(img.to(device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            with torch.autocast(device_type=device):
                outputs = model(images)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            res = {target["image_id"]: output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            train_coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            train_metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # gather the stats from all processes
        train_metric_logger.synchronize_between_processes()
        print("Training performance: ", train_metric_logger)
        train_coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        train_coco_evaluator.accumulate()
        train_coco_evaluator.summarize()


    val_coco_evaluator = CocoEvaluator(val_coco_ds, iou_types)

    # calculate validation accuracy
    for images, targets in val_metric_logger.log_every(val_data_loader, len(val_data_loader) // 2, 'Validation Accuracy: '):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        with torch.autocast(device_type=device):
            outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        val_coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        val_metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    val_metric_logger.synchronize_between_processes()
    print("Validation performance: ", val_metric_logger)
    val_coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    val_coco_evaluator.accumulate()
    val_coco_evaluator.summarize()

    torch.set_num_threads(n_threads)

    if train_data_loader is not None:
        return train_coco_evaluator, val_coco_evaluator
    else:
        return val_coco_evaluator