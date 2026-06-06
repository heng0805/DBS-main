features = {"V": "VST", "F": "I3D", "A": "AST"}

# v22 最终配置
dbs_config = {
    "batch_size": 32,
    "epoch": {"TES": 500, "PCS": 500, "TotalScore": 500},
    "num_workers": 4,
    
    "model_dim": 256,
    "K": 4,
    "fc_drop": 0.3,
    "fc_r": 2,
    "feat_drop": 0.4,
    "ms_heads": 4,
    "cm_heads": 4,
    "lfc_heads": 4,
    
    "in_dim": {"V": 1024, "F": 1024, "A": 768, "S1": 1024, "S2": 1024},
    "segments": 100,
    
    "ckpt_dir": "./checkpoints/dbs_v22",
    
    "optim": "AdamW",
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "reduce_fc_lr": False,

    "lr_scheduler": "cosine",
    "lr_min": 1e-6,
    "warmup_lr_init": 1e-4,    
    
    "clip_grad": True,
    "save_moniter": "rho",
    
    "use_lfc": True,
    "use_rcf": True,
    "lambda_coord": 0.2,
    "lambda_rhythm": 0.2,
}
