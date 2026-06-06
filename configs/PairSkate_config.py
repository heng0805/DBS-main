features = {"V": "VST", "F": "I3D", "A": "AST"}

pair_skate_v1 = {
    "batch_size": 16,
    "epoch": {"TES": 500, "PCS": 500, "TotalScore": 500},
    "num_workers": 4,
    
    "model_dim": 256,
    "K": 6,
    "fc_drop": 0.3,
    "fc_r": 2,
    "feat_drop": 0.5,
    "ms_heads": 4,
    "cm_heads": 4,
    "lfc_heads": 4,
    
    "in_dim": {"V": 1024, "F": 1024, "A": 768, "S1": 1024, "S2": 1024},
    "segments": 130,
    
    "ckpt_dir": "./checkpoints/pairskate_v1",
    
    "optim": "AdamW",
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "reduce_fc_lr": True,

    "lr_scheduler": "cosine",
    "lr_min": 1e-6,
    "warmup_lr_init": 1e-6,    
    
    "clip_grad": True,
    "save_moniter": "rho",
    
    "use_lfc": True,
    "use_rcf": True,
    "lambda_coord": 0.1,
    "lambda_rhythm": 0.1,
}

pair_skate_v2 = {
    "batch_size": 8,
    "epoch": {"TES": 500, "PCS": 500, "TotalScore": 500},
    "num_workers": 4,
    
    "model_dim": 512,
    "K": 8,
    "fc_drop": 0.4,
    "fc_r": 4,
    "feat_drop": 0.3,
    "ms_heads": 8,
    "cm_heads": 8,
    "lfc_heads": 8,
    
    "in_dim": {"V": 1024, "F": 1024, "A": 768, "S1": 1024, "S2": 1024},
    "segments": 150,
    
    "ckpt_dir": "./checkpoints/pairskate_v2",
    
    "optim": "AdamW",
    "lr": 3e-4,
    "weight_decay": 5e-4,
    "reduce_fc_lr": True,

    "lr_scheduler": "cosine",
    "lr_min": 1e-7,
    "warmup_lr_init": 1e-5,    
    
    "clip_grad": True,
    "save_moniter": "rho",
    
    "use_lfc": True,
    "use_rcf": True,
    "lambda_coord": 0.05,
    "lambda_rhythm": 0.05,
}

pair_skate_v3 = {
    "batch_size": 32,
    "epoch": {"TES": 500, "PCS": 500, "TotalScore": 500},
    "num_workers": 4,
    
    "model_dim": 256,
    "K": 4,
    "fc_drop": 0.2,
    "fc_r": 2,
    "feat_drop": 0.4,
    "ms_heads": 4,
    "cm_heads": 4,
    "lfc_heads": 4,
    
    "in_dim": {"V": 1024, "F": 1024, "A": 768, "S1": 1024, "S2": 1024},
    "segments": 100,
    
    "ckpt_dir": "./checkpoints/pairskate_v3",
    
    "optim": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "reduce_fc_lr": False,

    "lr_scheduler": "cosine",
    "lr_min": 1e-6,
    "warmup_lr_init": 1e-4,    
    
    "clip_grad": True,
    "save_moniter": "rho",
    
    "use_lfc": True,
    "use_rcf": True,
    "lambda_coord": 0.15,
    "lambda_rhythm": 0.15,
}

pair_skate_v4 = {
    "batch_size": 16,
    "epoch": {"TES": 500, "PCS": 500, "TotalScore": 500},
    "num_workers": 4,
    
    "model_dim": 384,
    "K": 6,
    "fc_drop": 0.35,
    "fc_r": 3,
    "feat_drop": 0.45,
    "ms_heads": 6,
    "cm_heads": 6,
    "lfc_heads": 6,
    
    "in_dim": {"V": 1024, "F": 1024, "A": 768, "S1": 1024, "S2": 1024},
    "segments": 130,
    
    "ckpt_dir": "./checkpoints/pairskate_v4",
    
    "optim": "AdamW",
    "lr": 4e-4,
    "weight_decay": 2e-4,
    "reduce_fc_lr": True,

    "lr_scheduler": "cosine",
    "lr_min": 5e-7,
    "warmup_lr_init": 5e-6,    
    
    "clip_grad": True,
    "save_moniter": "rho",
    
    "use_lfc": True,
    "use_rcf": True,
    "lambda_coord": 0.08,
    "lambda_rhythm": 0.08,
}

pair_skate_v5 = {
    "batch_size": 8,
    "epoch": {"TES": 500, "PCS": 500, "TotalScore": 500},
    "num_workers": 4,
    
    "model_dim": 512,
    "K": 10,
    "fc_drop": 0.35,
    "fc_r": 4,
    "feat_drop": 0.25,
    "ms_heads": 8,
    "cm_heads": 8,
    "lfc_heads": 8,
    
    "in_dim": {"V": 1024, "F": 1024, "A": 768, "S1": 1024, "S2": 1024},
    "segments": 160,
    
    "ckpt_dir": "./checkpoints/pairskate_v5",
    
    "optim": "AdamW",
    "lr": 2e-4,
    "weight_decay": 3e-4,
    "reduce_fc_lr": True,

    "lr_scheduler": "cosine",
    "lr_min": 1e-7,
    "warmup_lr_init": 1e-5,    
    
    "clip_grad": True,
    "save_moniter": "rho",
    
    "use_lfc": True,
    "use_rcf": True,
    "lambda_coord": 0.03,
    "lambda_rhythm": 0.03,
}

pair_skate_v6 = {
    "batch_size": 32,
    "epoch": {"TES": 500, "PCS": 500, "TotalScore": 500},
    "num_workers": 4,
    
    "model_dim": 320,
    "K": 5,
    "fc_drop": 0.18,
    "fc_r": 2,
    "feat_drop": 0.35,
    "ms_heads": 5,
    "cm_heads": 5,
    "lfc_heads": 5,
    
    "in_dim": {"V": 1024, "F": 1024, "A": 768, "S1": 1024, "S2": 1024},
    "segments": 110,
    
    "ckpt_dir": "./checkpoints/pairskate_v6",
    
    "optim": "AdamW",
    "lr": 8e-4,
    "weight_decay": 5e-6,
    "reduce_fc_lr": False,

    "lr_scheduler": "cosine",
    "lr_min": 5e-7,
    "warmup_lr_init": 5e-5,    
    
    "clip_grad": True,
    "save_moniter": "rho",
    
    "use_lfc": True,
    "use_rcf": True,
    "lambda_coord": 0.12,
    "lambda_rhythm": 0.12,
}

pair_skate_v7 = {
    "batch_size": 24,
    "epoch": {"TES": 500, "PCS": 500, "TotalScore": 500},
    "num_workers": 4,
    
    "model_dim": 256,
    "K": 4,
    "fc_drop": 0.22,
    "fc_r": 2,
    "feat_drop": 0.38,
    "ms_heads": 4,
    "cm_heads": 4,
    "lfc_heads": 4,
    
    "in_dim": {"V": 1024, "F": 1024, "A": 768, "S1": 1024, "S2": 1024},
    "segments": 105,
    
    "ckpt_dir": "./checkpoints/pairskate_v7",
    
    "optim": "AdamW",
    "lr": 9e-4,
    "weight_decay": 8e-6,
    "reduce_fc_lr": False,

    "lr_scheduler": "cosine",
    "lr_min": 8e-7,
    "warmup_lr_init": 8e-5,    
    
    "clip_grad": True,
    "save_moniter": "rho",
    
    "use_lfc": True,
    "use_rcf": True,
    "lambda_coord": 0.13,
    "lambda_rhythm": 0.13,
}

pair_skate_v8 = {
    "batch_size": 32,
    "epoch": {"TES": 500, "PCS": 500, "TotalScore": 500},
    "num_workers": 4,
    
    "model_dim": 384,
    "K": 5,
    "fc_drop": 0.25,
    "fc_r": 3,
    "feat_drop": 0.35,
    "ms_heads": 6,
    "cm_heads": 6,
    "lfc_heads": 6,
    
    "in_dim": {"V": 1024, "F": 1024, "A": 768, "S1": 1024, "S2": 1024},
    "segments": 115,
    
    "ckpt_dir": "./checkpoints/pairskate_v8",
    
    "optim": "AdamW",
    "lr": 7e-4,
    "weight_decay": 1e-5,
    "reduce_fc_lr": False,

    "lr_scheduler": "cosine",
    "lr_min": 6e-7,
    "warmup_lr_init": 6e-5,    
    
    "clip_grad": True,
    "save_moniter": "rho",
    
    "use_lfc": True,
    "use_rcf": True,
    "lambda_coord": 0.14,
    "lambda_rhythm": 0.14,
}

pair_skate_v9 = {
    "batch_size": 32,
    "epoch": {"TES": 500, "PCS": 500, "TotalScore": 500},
    "num_workers": 4,
    
    "model_dim": 256,
    "K": 4,
    "fc_drop": 0.3,
    "fc_r": 2,
    "feat_drop": 0.5,
    "ms_heads": 4,
    "cm_heads": 4,
    "lfc_heads": 4,
    
    "in_dim": {"V": 1024, "F": 1024, "A": 768, "S1": 1024, "S2": 1024},
    "segments": 100,
    
    "ckpt_dir": "./checkpoints/pairskate_v9",
    
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

pair_skate_v10 = {
    "batch_size": 32,
    "epoch": {"TES": 500, "PCS": 500, "TotalScore": 500},
    "num_workers": 4,
    
    "model_dim": 256,
    "K": 4,
    "fc_drop": 0.15,
    "fc_r": 2,
    "feat_drop": 0.35,
    "ms_heads": 4,
    "cm_heads": 4,
    "lfc_heads": 4,
    
    "in_dim": {"V": 1024, "F": 1024, "A": 768, "S1": 1024, "S2": 1024},
    "segments": 95,
    
    "ckpt_dir": "./checkpoints/pairskate_v10",
    
    "optim": "AdamW",
    "lr": 1.05e-3,
    "weight_decay": 5e-6,
    "reduce_fc_lr": False,

    "lr_scheduler": "cosine",
    "lr_min": 8e-7,
    "warmup_lr_init": 8e-5,    
    
    "clip_grad": True,
    "save_moniter": "rho",
    
    "use_lfc": True,
    "use_rcf": True,
    "lambda_coord": 0.18,
    "lambda_rhythm": 0.18,
}

pair_skate_v11 = {
    "batch_size": 28,
    "epoch": {"TES": 500, "PCS": 500, "TotalScore": 500},
    "num_workers": 4,
    
    "model_dim": 256,
    "K": 4,
    "fc_drop": 0.19,
    "fc_r": 2,
    "feat_drop": 0.42,
    "ms_heads": 4,
    "cm_heads": 4,
    "lfc_heads": 4,
    
    "in_dim": {"V": 1024, "F": 1024, "A": 768, "S1": 1024, "S2": 1024},
    "segments": 98,
    
    "ckpt_dir": "./checkpoints/pairskate_v11",
    
    "optim": "AdamW",
    "lr": 9.5e-4,
    "weight_decay": 7e-6,
    "reduce_fc_lr": False,

    "lr_scheduler": "cosine",
    "lr_min": 9e-7,
    "warmup_lr_init": 9e-5,    
    
    "clip_grad": True,
    "save_moniter": "rho",
    
    "use_lfc": True,
    "use_rcf": True,
    "lambda_coord": 0.17,
    "lambda_rhythm": 0.17,
}

pair_skate_v12 = {
    "batch_size": 32,
    "epoch": {"TES": 500, "PCS": 500, "TotalScore": 500},
    "num_workers": 4,
    
    "model_dim": 288,
    "K": 4,
    "fc_drop": 0.17,
    "fc_r": 2,
    "feat_drop": 0.36,
    "ms_heads": 4,
    "cm_heads": 4,
    "lfc_heads": 4,
    
    "in_dim": {"V": 1024, "F": 1024, "A": 768, "S1": 1024, "S2": 1024},
    "segments": 102,
    
    "ckpt_dir": "./checkpoints/pairskate_v12",
    
    "optim": "AdamW",
    "lr": 9.8e-4,
    "weight_decay": 6e-6,
    "reduce_fc_lr": False,

    "lr_scheduler": "cosine",
    "lr_min": 7e-7,
    "warmup_lr_init": 7e-5,    
    
    "clip_grad": True,
    "save_moniter": "rho",
    
    "use_lfc": True,
    "use_rcf": True,
    "lambda_coord": 0.155,
    "lambda_rhythm": 0.155,
}

pair_skate_p1 = {
    "batch_size": 32,
    "epoch": {"TES": 500, "PCS": 500, "TotalScore": 500},
    "num_workers": 4,
    
    "model_dim": 256,
    "K": 4,
    "fc_drop": 0.3,
    "fc_r": 2,
    "feat_drop": 0.5,
    "ms_heads": 4,
    "cm_heads": 4,
    "lfc_heads": 4,
    
    "in_dim": {"V": 1024, "F": 1024, "A": 768, "S1": 1024, "S2": 1024},
    "segments": 100,
    
    "ckpt_dir": "./checkpoints/pairskate_p1",
    
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

pair_skate_p2 = {
    "batch_size": 32,
    "epoch": {"TES": 500, "PCS": 500, "TotalScore": 500},
    "num_workers": 4,
    
    "model_dim": 256,
    "K": 4,
    "fc_drop": 0.16,
    "fc_r": 2,
    "feat_drop": 0.35,
    "ms_heads": 4,
    "cm_heads": 4,
    "lfc_heads": 4,
    
    "in_dim": {"V": 1024, "F": 1024, "A": 768, "S1": 1024, "S2": 1024},
    "segments": 105,
    
    "ckpt_dir": "./checkpoints/pairskate_p2",
    
    "optim": "AdamW",
    "lr": 1.05e-3,
    "weight_decay": 6e-6,
    "reduce_fc_lr": False,

    "lr_scheduler": "cosine",
    "lr_min": 8e-7,
    "warmup_lr_init": 8e-5,    
    
    "clip_grad": True,
    "save_moniter": "rho",
    
    "use_lfc": True,
    "use_rcf": True,
    "lambda_coord": 0.14,
    "lambda_rhythm": 0.14,
}

pair_skate_p3 = {
    "batch_size": 32,
    "epoch": {"TES": 500, "PCS": 500, "TotalScore": 500},
    "num_workers": 4,
    
    "model_dim": 256,
    "K": 4,
    "fc_drop": 0.2,
    "fc_r": 2,
    "feat_drop": 0.4,
    "ms_heads": 4,
    "cm_heads": 4,
    "lfc_heads": 4,
    
    "in_dim": {"V": 1024, "F": 1024, "A": 768, "S1": 1024, "S2": 1024},
    "segments": 100,
    
    "ckpt_dir": "./checkpoints/pairskate_p3",
    
    "optim": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "reduce_fc_lr": False,

    "lr_scheduler": "cosine",
    "lr_min": 1e-6,
    "warmup_lr_init": 1e-4,    
    
    "clip_grad": True,
    "save_moniter": "rho",
    
    "use_lfc": True,
    "use_rcf": True,
    "lambda_coord": 0.12,
    "lambda_rhythm": 0.12,
}

pair_skate_p4 = {
    "batch_size": 32,
    "epoch": {"TES": 500, "PCS": 500, "TotalScore": 500},
    "num_workers": 4,
    
    "model_dim": 288,
    "K": 4,
    "fc_drop": 0.17,
    "fc_r": 2,
    "feat_drop": 0.36,
    "ms_heads": 4,
    "cm_heads": 4,
    "lfc_heads": 4,
    
    "in_dim": {"V": 1024, "F": 1024, "A": 768, "S1": 1024, "S2": 1024},
    "segments": 102,
    
    "ckpt_dir": "./checkpoints/pairskate_p4",
    
    "optim": "AdamW",
    "lr": 1.08e-3,
    "weight_decay": 7e-6,
    "reduce_fc_lr": False,

    "lr_scheduler": "cosine",
    "lr_min": 9e-7,
    "warmup_lr_init": 9e-5,    
    
    "clip_grad": True,
    "save_moniter": "rho",
    
    "use_lfc": True,
    "use_rcf": True,
    "lambda_coord": 0.15,
    "lambda_rhythm": 0.15,
}

pair_skate_p5 = {
    "batch_size": 32,
    "epoch": {"TES": 500, "PCS": 500, "TotalScore": 500},
    "num_workers": 4,
    
    "model_dim": 256,
    "K": 4,
    "fc_drop": 0.15,
    "fc_r": 2,
    "feat_drop": 0.32,
    "ms_heads": 4,
    "cm_heads": 4,
    "lfc_heads": 4,
    
    "in_dim": {"V": 1024, "F": 1024, "A": 768, "S1": 1024, "S2": 1024},
    "segments": 98,
    
    "ckpt_dir": "./checkpoints/pairskate_p5",
    
    "optim": "AdamW",
    "lr": 1.12e-3,
    "weight_decay": 5e-6,
    "reduce_fc_lr": False,

    "lr_scheduler": "cosine",
    "lr_min": 1e-6,
    "warmup_lr_init": 1e-4,    
    
    "clip_grad": True,
    "save_moniter": "rho",
    
    "use_lfc": True,
    "use_rcf": True,
    "lambda_coord": 0.13,
    "lambda_rhythm": 0.13,
}

pair_skate = pair_skate_v9
