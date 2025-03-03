import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import segmentation_pytorch.utils
from segmentation_pytorch.coco_utils import get_coco_api_from_dataset
from segmentation_pytorch.coco_eval import CustomCocoEvaluator

##################################################################
 #######             Semantic Segmentation             ########## 
##################################################################

# def train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq, accumulation_steps, val_data_loader=None):
def train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq, accumulation_steps, val_data_loader=None, step_epoch_counter=None):
    """
    Train model for one epoch on training dataset and return training loss to logger.
    
    :param model: torch.nn.Module
    :param optimizer: torch.optim.Optimizer
    :param train_data_loader: torch.utils.data.DataLoader
    :param device: torch.device
    :param epoch: int
    :param print_freq: int
    :param accumulation_steps: int
    :param val_data_loader: torch.utils.data.DataLoader (default=None)
    :return: train_metric_logger, val_metric_logger (if val_data_loader is not None)
    """
    model.train()
    train_metric_logger = segmentation_pytorch.utils.MetricLogger(delimiter="  ")
    train_metric_logger.add_meter('lr', segmentation_pytorch.utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    train_metric_logger.add_meter('total_loss', segmentation_pytorch.utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    train_metric_logger.add_meter('bce_loss', segmentation_pytorch.utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    train_metric_logger.add_meter('dice_loss', segmentation_pytorch.utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    train_metric_logger.add_meter('sup3_loss', segmentation_pytorch.utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    train_metric_logger.add_meter('sup2_loss', segmentation_pytorch.utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    train_metric_logger.add_meter('sup1_loss', segmentation_pytorch.utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    train_header = 'Epoch: [{}] Training'.format(epoch)

    lr_scheduler = None
    # Apply warmup at the start of each training step
    if step_epoch_counter == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(train_data_loader) // accumulation_steps - 1)
        lr_scheduler = segmentation_pytorch.utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    
    # Initialize gradient scaler for mixed precision training
    scaler = torch.GradScaler(device)

    for i, (images, targets) in enumerate(train_metric_logger.log_every(train_data_loader, print_freq, train_header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        if (i % accumulation_steps == 0):
            optimizer.zero_grad()

        with torch.autocast(device_type=device):
            train_loss_dict = model(images, targets)
            train_losses = train_loss_dict['total_loss'] / accumulation_steps

        train_loss_dict_reduced = segmentation_pytorch.utils.reduce_dict(train_loss_dict)
        train_losses_reduced = train_loss_dict_reduced['total_loss']
        train_loss_value = train_losses_reduced.item()

        if not math.isfinite(train_loss_value):
            print("Loss is {}, stopping training".format(train_loss_value))
            print(train_loss_dict_reduced)
            sys.exit(1)

        scaler.scale(train_losses).backward()

        if ((i + 1) % accumulation_steps == 0) or (i + 1 == len(train_data_loader)):
            scaler.step(optimizer)
            scaler.update()
            
            if lr_scheduler is not None:
                lr_scheduler.step()

        # Update all loss metrics
        train_metric_logger.update(
            total_loss=train_loss_dict_reduced['total_loss'].item(),
            bce_loss=train_loss_dict_reduced['bce_loss'].item(),
            dice_loss=train_loss_dict_reduced['dice_loss'].item(),
            sup3_loss=train_loss_dict_reduced['sup3_loss'].item(),
            sup2_loss=train_loss_dict_reduced['sup2_loss'].item(),
            sup1_loss=train_loss_dict_reduced['sup1_loss'].item()
        )
        train_metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # validation loop
    if val_data_loader is not None:
        val_metric_logger = segmentation_pytorch.utils.MetricLogger(delimiter="  ")
        val_metric_logger.add_meter('lr', segmentation_pytorch.utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        val_metric_logger.add_meter('total_loss', segmentation_pytorch.utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
        val_metric_logger.add_meter('bce_loss', segmentation_pytorch.utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
        val_metric_logger.add_meter('dice_loss', segmentation_pytorch.utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
        val_metric_logger.add_meter('sup3_loss', segmentation_pytorch.utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
        val_metric_logger.add_meter('sup2_loss', segmentation_pytorch.utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
        val_metric_logger.add_meter('sup1_loss', segmentation_pytorch.utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
        val_header = 'Epoch: [{}] Validation'.format(epoch)

        with torch.no_grad():
            for images, targets in val_metric_logger.log_every(val_data_loader, len(val_data_loader) // 4, val_header):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

                with torch.autocast(device_type=device):
                    val_loss_dict = model(images, targets)

                val_loss_dict_reduced = segmentation_pytorch.utils.reduce_dict(val_loss_dict)
                val_losses_reduced = val_loss_dict_reduced['total_loss']
                val_loss_value = val_losses_reduced.item()

                if not math.isfinite(val_loss_value):
                    print("Loss is {}, stopping training".format(val_loss_value))
                    print(val_loss_dict_reduced)
                    sys.exit(1)

                # Update all validation loss metrics
                val_metric_logger.update(
                    total_loss=val_loss_dict_reduced['total_loss'].item(),
                    bce_loss=val_loss_dict_reduced['bce_loss'].item(),
                    dice_loss=val_loss_dict_reduced['dice_loss'].item(),
                    sup3_loss=val_loss_dict_reduced['sup3_loss'].item(),
                    sup2_loss=val_loss_dict_reduced['sup2_loss'].item(),
                    sup1_loss=val_loss_dict_reduced['sup1_loss'].item()
                )
                val_metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    if val_data_loader is not None:
        return train_metric_logger, val_metric_logger
    else:
        return train_metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
        
    # For semantic segmentation, we only need segmentation IoU
    if model_without_ddp.__class__.__name__ == 'SmallObjectUNet':
        return ["segm"]
    
    # Fallback to original type checking
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
        train_coco_evaluator = CustomCocoEvaluator(train_coco_ds, iou_types)

        # calculate training accuracy
        for images, targets in train_metric_logger.log_every(train_data_loader, len(train_data_loader) // 4, 'Training Accuracy: '):
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
        print()
        train_coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        train_coco_evaluator.accumulate()
        train_coco_evaluator.summarize()

    val_coco_evaluator = CustomCocoEvaluator(val_coco_ds, iou_types)

    # calculate validation accuracy
    for images, targets in val_metric_logger.log_every(val_data_loader, len(val_data_loader) // 4, 'Validation Accuracy: '):
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
    print()
    val_coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    val_coco_evaluator.accumulate()
    val_coco_evaluator.summarize()

    torch.set_num_threads(n_threads)

    if train_data_loader is not None:
        return train_coco_evaluator, val_coco_evaluator
    else:
        return val_coco_evaluator