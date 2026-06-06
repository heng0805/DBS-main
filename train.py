import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from pipeline import go_train, log_info
from functools import partial
import configs.DBS_config as conf


def setup_seed(seed):
    import numpy as np
    import torch
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


def train(gpu='0', score_type='tes_score'):
    cur_conf = conf.dbs_config
    
    if gpu != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    setup_seed(0)
    
    epochs = cur_conf["epoch"].get(score_type.replace('_score', '').upper(), 500)
    batch_size = cur_conf["batch_size"]
    lr = cur_conf["lr"]
    
    dataset_config = {
        "batch_size": batch_size,
        "num_workers": cur_conf["num_workers"],
        "segments": cur_conf.get("segments", 100)
    }
    
    model_config = {
        "model_name": "pamfn_pairskate",
        "model_conf": {
            "model_dim": cur_conf["model_dim"],
            "fc_drop": cur_conf.get("fc_drop", 0),
            "fc_r": cur_conf.get("fc_r", 2),
            "feat_drop": cur_conf.get("feat_drop", 0.5),
            "K": cur_conf.get("K", 4),
            "ms_heads": cur_conf.get("ms_heads", 4),
            "cm_heads": cur_conf.get("cm_heads", 4),
            "lfc_heads": cur_conf.get("lfc_heads", 4),
            "in_dim": cur_conf.get("in_dim", {"V": 1024, "F": 1024, "A": 768, "S1": 1024, "S2": 1024}),
            "use_lfc": cur_conf.get("use_lfc", True),
            "use_rcf": cur_conf.get("use_rcf", True),
            "lambda_coord": cur_conf.get("lambda_coord", 0.1),
            "lambda_rhythm": cur_conf.get("lambda_rhythm", 0.1),
        }
    }
    
    optimizer_config = {
        "optim": cur_conf["optim"],
        "lr": lr,
        "weight_decay": cur_conf["weight_decay"],
        "reduce_fc_lr": cur_conf.get("reduce_fc_lr", False)
    }
    
    lr_scheduler = {
        "scheduler": cur_conf["lr_scheduler"],
        "epoch_num": epochs,
        "scheduler_steps": cur_conf["epoch"],
        "lr_min": cur_conf["lr_min"],
        "warmup_lr_init": cur_conf["warmup_lr_init"],
        "decay_rate": 0.1
    }
    
    log_func = partial(log_info, logger=None, print_log=True)
    
    ckpt_dir = cur_conf.get("ckpt_dir", f"./checkpoints/dbs_{score_type.replace('_score', '')}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    save_base_model = os.path.join(ckpt_dir, f"dbs_{score_type}_best.pth")
    save_final_model = os.path.join(ckpt_dir, f"dbs_{score_type}_final.pth")
    
    print(f"=" * 60)
    print(f"训练配置 (v22):")
    print(f"  - 评分类型: {score_type}")
    print(f"  - 训练轮数: {epochs}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 学习率: {lr}")
    print(f"  - fc_drop: {cur_conf['fc_drop']}")
    print(f"  - feat_drop: {cur_conf['feat_drop']}")
    print(f"  - weight_decay: {cur_conf['weight_decay']}")
    print(f"  - lambda: {cur_conf['lambda_coord']}")
    print(f"  - Segments: {cur_conf['segments']}")
    print(f"  - GPU: {gpu}")
    print(f"  - 模型保存路径: {ckpt_dir}")
    print(f"=" * 60)
    
    ret = go_train(
        model_config, 
        optimizer_config, 
        lr_scheduler, 
        dataset_config, 
        epochs,
        save_base_model=save_base_model, 
        save_moniter=cur_conf["save_moniter"], 
        save_final_model=save_final_model,
        log_func=log_func, 
        seed=0, 
        clip_grad=cur_conf["clip_grad"],
        score_type=score_type
    )
    
    print(f"\n训练完成!")
    print(f"Best Spearman: {ret[0]/100:.4f}")
    print(f"Saved Model Spearman: {ret[1]/100:.4f}")
    print(f"Best R-L2: {ret[2]:.4f}")
    print(f"Saved Model R-L2: {ret[3]:.4f}")
    
    return ret


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train DBS Model (v22)')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--score_type', type=str, default='tes_score', 
                        choices=['total_score', 'tes_score', 'pcs_score'],
                        help='Score type to predict')
    
    args = parser.parse_args()
    
    train(
        gpu=args.gpu,
        score_type=args.score_type
    )
