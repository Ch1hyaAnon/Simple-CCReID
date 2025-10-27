import time
import datetime
import logging
import torch
# from apex import amp  <-- 1. 删除这一行
from torch.cuda.amp import autocast # <-- 2. 添加这一行
from tools.utils import AverageMeter


# 3. 在函数定义中添加 scaler 参数
def train_cal(config, epoch, model, classifier, clothes_classifier, criterion_cla, criterion_pair, 
    criterion_clothes, criterion_adv, optimizer, optimizer_cc, trainloader, pid2clothes, scaler):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_clo_loss = AverageMeter()
    batch_adv_loss = AverageMeter()
    corrects = AverageMeter()
    clothes_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()
    clothes_classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()
        # Measure data loading time
        data_time.update(time.time() - end)

        # -----------------------------------------------------------------
        # 4. 修改训练逻辑以使用 torch.cuda.amp
        # -----------------------------------------------------------------

        # --- Part 1: 更新衣服分类器 (Update the clothes discriminator) ---
        optimizer_cc.zero_grad()
        
        # 使用 autocast 上下文管理器进行前向传播和损失计算
        with autocast(enabled=config.TRAIN.AMP):
            features = model(imgs)
            pred_clothes = clothes_classifier(features.detach())
            clothes_loss = criterion_clothes(pred_clothes, clothes_ids)

        if epoch >= config.TRAIN.START_EPOCH_CC:
            # 使用 scaler 缩放损失并进行反向传播
            scaler.scale(clothes_loss).backward()
            # 使用 scaler 更新优化器
            scaler.step(optimizer_cc)

        # --- Part 2: 更新主干网络 (Update the backbone) ---
        optimizer.zero_grad()
        
        # 再次使用 autocast
        with autocast(enabled=config.TRAIN.AMP):
            # 注意: features 已经计算过, 但为了梯度流, 可能需要重新计算或直接使用
            # 这里的代码逻辑是正确的, 因为 clothes_classifier 需要新的、带梯度的 features
            new_pred_clothes = clothes_classifier(features) 
            outputs = classifier(features)

            # 计算损失
            cla_loss = criterion_cla(outputs, pids)
            pair_loss = criterion_pair(features, pids)
            adv_loss = criterion_adv(new_pred_clothes, clothes_ids, pos_mask)
            if epoch >= config.TRAIN.START_EPOCH_ADV:
                loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss   
            else:
                loss = cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss   
        
        # 使用 scaler 缩放损失并进行反向传播
        scaler.scale(loss).backward()
        # 使用 scaler 更新优化器
        scaler.step(optimizer)

        # --- Part 3: 更新 scaler ---
        # 在所有优化器步骤完成后, 更新 scaler 的状态
        scaler.update()
        
        # -----------------------------------------------------------------
        # (修改结束)
        # -----------------------------------------------------------------
        
        # statistics (这部分无需修改)
        _, preds = torch.max(outputs.data, 1)
        _, clothes_preds = torch.max(new_pred_clothes.data, 1)
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        clothes_corrects.update(torch.sum(clothes_preds == clothes_ids.data).float()/clothes_ids.size(0), clothes_ids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_clo_loss.update(clothes_loss.item(), clothes_ids.size(0))
        batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                  'Time:{batch_time.sum:.1f}s '
                  'Data:{data_time.sum:.1f}s '
                  'ClaLoss:{cla_loss.avg:.4f} '
                  'PairLoss:{pair_loss.avg:.4f} '
                  'CloLoss:{clo_loss.avg:.4f} '
                  'AdvLoss:{adv_loss.avg:.4f} '
                  'Acc:{acc.avg:.2%} '
                  'CloAcc:{clo_acc.avg:.2%} '.format(
                   epoch+1, batch_time=batch_time, data_time=data_time, 
                   cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, 
                   clo_loss=batch_clo_loss, adv_loss=batch_adv_loss, 
                   acc=corrects, clo_acc=clothes_corrects))


# 3. 在函数定义中添加 scaler 参数
def train_cal_with_memory(config, epoch, model, classifier, criterion_cla, criterion_pair, 
    criterion_adv, optimizer, trainloader, pid2clothes, scaler):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_adv_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        
        optimizer.zero_grad()
        
        # 4. 使用 autocast 包裹前向传播和损失计算
        with autocast(enabled=config.TRAIN.AMP):
            # Forward
            features = model(imgs)
            outputs = classifier(features)
            
            # Compute loss
            cla_loss = criterion_cla(outputs, pids)
            pair_loss = criterion_pair(features, pids)

            if epoch >= config.TRAIN.START_EPOCH_ADV:
                adv_loss = criterion_adv(features, clothes_ids, pos_mask)
                loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss   
            else:
                loss = cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss  

        # 5. 使用新的三步法进行反向传播和优化
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # statistics (这部分无需修改)
        _, preds = torch.max(outputs.data, 1)
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        if epoch >= config.TRAIN.START_EPOCH_ADV: 
            batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                'Time:{batch_time.sum:.1f}s '
                'Data:{data_time.sum:.1f}s '
                'ClaLoss:{cla_loss.avg:.4f} '
                'PairLoss:{pair_loss.avg:.4f} '
                'AdvLoss:{adv_loss.avg:.4f} '
                'Acc:{acc.avg:.2%} '.format(
                epoch+1, batch_time=batch_time, data_time=data_time, 
                cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, 
                adv_loss=batch_adv_loss, acc=corrects))