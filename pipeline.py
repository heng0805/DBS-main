from typing import List, Tuple

import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from datasets import PairSkateDataset
from utils import Logger, AverageMeter, now_to_date
from utils.lr_scheduler import build_scheduler


def build_pair_skate_dataset(batch_size=16, num_workers=4, segments=130):
    train_dset = PairSkateDataset(is_train=True, segments=segments)
    test_dset = PairSkateDataset(is_train=False, segments=segments)
    
    train_dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True,
                               pin_memory=False, num_workers=num_workers, drop_last=True)
    test_dloader = DataLoader(test_dset, batch_size=1, shuffle=False,
                              pin_memory=False, num_workers=num_workers)
    return train_dloader, test_dloader


def build_model(model_name, model_conf=None):
    if model_name == "pamfn_pairskate":
        from models.dbs_pairskate import DBSModel
        return DBSModel(**model_conf)
    elif model_name == "dbs_pairskate":
        from models.dbs_pairskate import DBSModel
        return DBSModel(**model_conf)
    else:
        raise ValueError(f"model: {model_name} is not supported!")


def build_optimizer(model, optim, lr, momentum=0.9, weight_decay=None, reduce_fc_lr=False, model_name=None):
    if reduce_fc_lr and model_name == 'pamfn_final':
        lr_scale = 0.1
        fc_params = list(map(id, model.fc_main.parameters())) + \
                    list(map(id, model.fc_aux_cls.parameters())) + \
                    list(map(id, model.fc_aux_reg.parameters()))
        other_params = filter(lambda p: id(p) not in fc_params, model.parameters())
        params_group = [
            {'params': model.fc_main.parameters(), 'lr': lr * lr_scale},
            {'params': model.fc_aux.parameters(), 'lr': lr * lr_scale},
            {'params': model.fc_aux_reg.parameters(), 'lr': lr * lr_scale},
            {'params': other_params}
        ]
    else:
        params_group = model.parameters()

    if optim == 'Adam':
        optimizer = torch.optim.Adam(params_group, lr=lr)
    elif optim == 'SGD':
        optimizer = torch.optim.SGD(params_group, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim == 'AdamW':
        optimizer = torch.optim.AdamW(params_group, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"optimizer: {optim} not found!")
    return optimizer


def build_lr_scheduler(optimizer, scheduler, epoch_num, scheduler_steps,
                    lr_min=5e-6, warmup_lr_init=5e-6, decay_rate=0.1):
    return build_scheduler(optimizer, scheduler, epoch_num, scheduler_steps, decay_rate=decay_rate,
                    lr_min=lr_min, warmup_lr_init=warmup_lr_init)


def train_epoch(model, optimizer, lr_scheduler, dataloader, cur_epoch, clip_grad=False, score_type='total_score'):
    losses = AverageMeter()
    total_preds = []
    total_labels = []
    tm = time.time()

    model.train()

    for i, batch in enumerate(dataloader):
        rgb_data = batch['rgb'].cuda()
        flow_data = batch['flow'].cuda()
        audio_data = batch['audio'].cuda()
        seq1_data = batch['seq1'].cuda()
        seq2_data = batch['seq2'].cuda()
        total_score = batch['total_score'].float().cuda()
        pcs_score = batch['pcs_score'].float().cuda()
        tes_score = batch['tes_score'].float().cuda()
        video_id = batch['video_id']
        
        if score_type == 'total_score':
            label = total_score
        elif score_type == 'pcs_score':
            label = pcs_score
        elif score_type == 'tes_score':
            label = tes_score
        else:
            label = total_score

        data_dict = {
            "V": rgb_data,
            "F": flow_data,
            "A": audio_data,
            "S1": seq1_data,
            "S2": seq2_data
        }
        
        preds, other_info = model(data_dict)
        loss = model.call_loss(preds, label, **other_info)

        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()

        losses.update(loss.item(), len(label))
        
        preds_squeezed = preds.squeeze()
        preds_list = preds_squeezed.cpu().detach().numpy().tolist()
        if isinstance(preds_list, float):
            total_preds.append(preds_list)
        else:
            total_preds.extend(preds_list)
        
        total_labels.extend(label.cpu().detach().numpy().tolist())

    if lr_scheduler is not None:
        lr_scheduler.step(cur_epoch+1)

    tm = time.time() - tm
    
    coef, _ = spearmanr(total_preds, total_labels)
    if isinstance(coef, np.ndarray):
        coef = coef.item()

    return tm, losses.avg * 100, coef


def test_epoch(model, dataloader, score_type='total_score'):
    losses = AverageMeter()
    total_preds = []
    total_labels = []
    tm = time.time()

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            rgb_data = batch['rgb'].cuda()
            flow_data = batch['flow'].cuda()
            audio_data = batch['audio'].cuda()
            seq1_data = batch['seq1'].cuda()
            seq2_data = batch['seq2'].cuda()
            total_score = batch['total_score'].float().cuda()
            pcs_score = batch['pcs_score'].float().cuda()
            tes_score = batch['tes_score'].float().cuda()
            
            if score_type == 'total_score':
                label = total_score
            elif score_type == 'pcs_score':
                label = pcs_score
            elif score_type == 'tes_score':
                label = tes_score
            else:
                label = total_score

            data_dict = {
                "V": rgb_data,
                "F": flow_data,
                "A": audio_data,
                "S1": seq1_data,
                "S2": seq2_data
            }
            
            preds, other_info = model(data_dict)
            loss = model.call_loss(preds, label, **other_info)

            losses.update(loss.item(), label.shape[0])
            
            preds_squeezed = preds.squeeze()
            preds_list = preds_squeezed.cpu().detach().numpy().tolist()
            if isinstance(preds_list, float):
                total_preds.append(preds_list)
            else:
                total_preds.extend(preds_list)
            
            total_labels.extend(label.cpu().detach().numpy().tolist())

    coef, _ = spearmanr(total_preds, total_labels)
    
    score_max = {
        'total_score': 100.0,
        'tes_score': 45.0,
        'pcs_score': 45.0
    }.get(score_type, 100.0)
    
    original_preds = np.array(total_preds) * score_max
    original_labels = np.array(total_labels) * score_max
    
    rmse = np.sqrt(np.mean((original_preds - original_labels) ** 2))
    mean_label = np.mean(np.abs(original_labels))
    relative_l2 = rmse / mean_label if mean_label > 0 else 0.0

    tm = time.time() - tm

    return tm, losses.avg * 100, coef * 100, relative_l2


def train_loop(model, optimizer, lr_scheduler, train_dataloader, test_dataloader, epoch_num,
               save_base_model="", save_final_model="", save_moniter="",
               log_func=None, clip_grad=False, score_type='total_score'):
    max_coef = 0
    coef_of_saved_model = 0
    best_relative_l2 = float('inf')
    relative_l2_of_saved_model = 0
    _train_loss, _test_loss = [], []
    train_coefs, test_coefs = [], []
    
    for cur_epoch in range(epoch_num):
        if hasattr(model, 'update_epoch'):
            model.update_epoch(cur_epoch)
            
        train_tm, train_loss, train_coef = train_epoch(model, optimizer, lr_scheduler, train_dataloader,
                                                       cur_epoch, clip_grad=clip_grad, score_type=score_type)
        test_tm, test_loss, test_coef, relative_l2 = test_epoch(model, test_dataloader, score_type=score_type)

        train_coefs.append(train_coef), test_coefs.append(test_coef)
        _train_loss.append(train_loss), _test_loss.append(test_loss) 
        
        if save_base_model != '':
            if save_moniter == "rho" and test_coef > max_coef:
                torch.save(model.state_dict(), save_base_model)
                coef_of_saved_model = test_coef
                relative_l2_of_saved_model = relative_l2
        if save_final_model != "" and test_coef > max_coef:
            torch.save(model.state_dict(), save_final_model)

        max_coef = max(max_coef, test_coef)
        best_relative_l2 = min(best_relative_l2, relative_l2)
        
        log_func('Epoch[{0}/{1}] \t'
                 'Time: {train_tm:.1f}/{test_tm:.1f} \t'
                 'Loss {train_loss:.4f}/{test_loss:.4f} \t'
                 'Coef {train_coef:.2f}/{test_coef:.2f} \t'
                 'R-L2 {relative_l2:.4f} \t'
                 'BestCoef {max_coef:.2f}'.format(
            cur_epoch, epoch_num, train_tm=train_tm, test_tm=test_tm,
            train_loss=train_loss, test_loss=test_loss,
            train_coef=float(train_coef),
            test_coef=float(test_coef),
            relative_l2=relative_l2,
            max_coef=float(max_coef)
        ))

    return max_coef, coef_of_saved_model, best_relative_l2, relative_l2_of_saved_model


def go_train(model_kwargs, optimizer_kwargs, lr_scheduler_kwargs, dataset_kwargs, epoch_num,
         save_base_model="", save_final_model="", save_moniter="",
         log_func=None, seed=0, clip_grad=False, score_type='total_score'):
    setup_seed(seed)
    
    train_dloader, test_dloader = build_pair_skate_dataset(
        batch_size=dataset_kwargs.get('batch_size', 16),
        num_workers=dataset_kwargs.get('num_workers', 4),
        segments=dataset_kwargs.get('segments', 130)
    )

    model = build_model(**model_kwargs)
    optimizer = build_optimizer(model, model_name=model_kwargs["model_name"], **optimizer_kwargs)
    lr_scheduler = build_lr_scheduler(optimizer, **lr_scheduler_kwargs)

    model.cuda()
    rets = train_loop(model, optimizer, lr_scheduler, train_dloader, test_dloader, epoch_num=epoch_num, 
            save_base_model=save_base_model, save_moniter=save_moniter, save_final_model=save_final_model, 
            log_func=log_func, clip_grad=clip_grad, score_type=score_type)
    log_func(f"Best Spearman: {rets[0]/100:.4f}\t"
             f"Saved Model Spearman: {rets[1]/100:.4f}\t"
             f"Best R-L2: {rets[2]:.4f}\t"
             f"Saved Model R-L2: {rets[3]:.4f}")
    return rets


def test_model(model_kwargs, dataset_kwargs, ckpt_path="", log_func=None, seed=0, score_type='total_score'):
    setup_seed(seed)
    _, test_dloader = build_pair_skate_dataset(
        batch_size=dataset_kwargs.get('batch_size', 16),
        num_workers=dataset_kwargs.get('num_workers', 4),
        segments=dataset_kwargs.get('segments', 130)
    )

    model = build_model(**model_kwargs)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    
    model.cuda()

    rets = test_epoch(model, test_dloader, score_type=score_type)

    log_func(f"Spearman: {rets[2] / 100:.4f}")


def setup_seed(seed):
    import os
    import random
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False
    
    try:
        torch.use_deterministic_algorithms(True)
    except:
        pass  


def log_info(info, logger, print_log):
    if logger is not None:
        logger.info(info)
    elif print_log is not False:
        print(info)
