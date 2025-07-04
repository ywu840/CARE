import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
# from torch.cuda import amp
import torch.distributed as dist
import collections
from torch.nn import functional as F
from loss.supcontrast import SupConLoss

def do_train_stage1(cfg,
             model,
             train_loader_stage1,
             optimizer,
             scheduler,
             local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD 

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  

    loss_meter = AverageMeter()
    scaler = torch.amp.GradScaler()
    xent = SupConLoss(device)

    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))

    # Add time tracking variables
    epoch_start_time = time.monotonic()
    moving_avg_epoch_time = None

    image_features = []
    labels = []
    with torch.no_grad():
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1):
            img = img.to(device)
            target = vid.to(device)
            with torch.amp.autocast("cuda", enabled=True):
                image_feature = model(img, target, get_image = True)
                for i, img_feat in zip(target, image_feature):
                    labels.append(i)
                    image_features.append(img_feat.cpu())

        labels_list = torch.stack(labels, dim=0).cuda()    # [training_size]
        image_features_list = torch.stack(image_features, dim=0).cuda()    # [training_size, 512]

        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
        num_image = labels_list.shape[0]
        i_ter = num_image // batch
    del labels, image_features

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        # Update epoch start time
        epoch_start_time = time.monotonic()

        iter_list = torch.randperm(num_image).to(device)
        for i in range(i_ter+1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i*batch:(i+1)* batch]
            else:
                b_list = iter_list[i*batch - 1:num_image]
            
            target = labels_list[b_list]
            image_features = image_features_list[b_list]  # (batch_size, 512)
            
            with torch.amp.autocast("cuda", enabled=True):
                text_features = model(label = target, get_text = True, 
                                      get_image = True, img_features = image_features)  # (batch_size, ctx_dim)
            
            
            loss_i2t = xent(image_features, text_features, target, target)
            loss_t2i = xent(text_features, image_features, target, target)
            #print(f"loss_i2t: {loss_i2t}")
            #print(f"loss_t2i: {loss_t2i}\n")

            loss = loss_i2t + loss_t2i
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), len(train_loader_stage1),
                                    loss_meter.avg, scheduler._get_lr(epoch)[0]))

        # Calculate and log time estimates at the end of each epoch
        epoch_time = time.monotonic() - epoch_start_time
        if moving_avg_epoch_time is None:
            moving_avg_epoch_time = epoch_time
        else:
            moving_avg_epoch_time = 0.9 * moving_avg_epoch_time + 0.1 * epoch_time
        
        remaining_epochs = epochs - epoch
        estimated_time_left = timedelta(seconds=moving_avg_epoch_time * remaining_epochs)
        logger.info(f"Epoch {epoch} completed in {timedelta(seconds=epoch_time)}")
        logger.info(f"Estimated time remaining: {estimated_time_left}\n")


        if epoch % checkpoint_period == 0:
            seed = str(cfg.SOLVER.SEED)
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}_S{}.pth'.format(epoch, seed)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}_S{}.pth'.format(epoch, seed)))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))
