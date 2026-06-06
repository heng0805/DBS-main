import os
import json
import argparse
import numpy as np
from functools import partial
import importlib.util


from pipeline import go, log_info, test_model


def load_module_from_file(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', required=True)

    parser.add_argument('--feats', type=int, default=1, required=True)
    parser.add_argument('--action', type=str, default='Ball', required=True)
    parser.add_argument('--multi_modality', action='store_true', default=False)
    parser.add_argument('--modality', type=str, default='V')
    
    parser.add_argument('--feat_dir', type=str, default='./datasets')
    parser.add_argument('--ckpt_path', type=str, default='')
    
    parser.add_argument('--save_base_model', type=str, default="")
    parser.add_argument('--save_final_model', type=str, default="")
    
    parser.add_argument('--test', action='store_true', default=False)
    
    parser.add_argument('--print_flops', action='store_true', default=False)
    parser.add_argument('--auxiliary_classes', type=str, nargs='+', default=None,
                        help='List of auxiliary classes for dual-branch training')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_args()

    if args.gpu != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    if args.action in ["Ball", "Clubs", "Hoop", "Ribbon"]:
        if args.feats == 1:
            import configs.RG_feats1 as conf
        elif args.feats == 2:
            import configs.RG_feats2 as conf
        else:
            raise ValueError(f"feats-{args.feats} is not supported!")
    elif args.action in ["TES", "PCS"]:
        if args.feats == 1:
            import configs.FISV_feats1 as conf
        elif args.feats == 2:
            import configs.FISV_feats2 as conf
        else:
            raise ValueError(f"feats-{args.feats} is not supported!")
    elif args.action == "PairSkate":
        import configs.PairSkate_config as conf


    if args.multi_modality:
        cur_conf = conf.multi_modality
    elif args.action == "PairSkate":
        cur_conf = conf.pair_skate
    else:
        cur_conf = conf.single_modality

    dataset_config = {
        "dataset_name": args.action,
        "feat_dir": args.feat_dir,
        "rgb_feat": conf.features["V"],
        "flow_feat": conf.features["F"],
        "audio_feat": conf.features["A"],
        "batch_size": cur_conf["batch_size"],
        "num_workers": cur_conf["num_workers"],
        "main_class": args.action,  
        "auxiliary_classes": args.auxiliary_classes,
        "segments": cur_conf.get("segments", -1)
    }

    if args.action == "PairSkate":
        if args.multi_modality:
            model_config = {
                "model_name": "pamfn_pairskate",
                "model_conf": {
                    "model_dim": cur_conf["model_dim"],
                    "fc_drop": cur_conf.get("fc_drop", 0),
                    "fc_r": cur_conf.get("fc_r", 2),
                    "feat_drop": cur_conf.get("feat_drop", 0.5),
                    "K": cur_conf.get("K", 6),
                    "ms_heads": cur_conf.get("ms_heads", 1),
                    "cm_heads": cur_conf.get("cm_heads", 1),
                    "lfc_heads": cur_conf.get("lfc_heads", 1),
                    "in_dim": cur_conf.get("in_dim", {"V": 1024, "F": 1024, "A": 768, "S1": 1024, "S2": 1024}),
                    "use_lfc": cur_conf.get("use_lfc", True),
                    "use_rcf": cur_conf.get("use_rcf", True),
                    "lambda_coord": cur_conf.get("lambda_coord", 0.1),
                    "lambda_rhythm": cur_conf.get("lambda_rhythm", 0.1),
                    "dataset_name": args.action,
                    "ckpt_dir": cur_conf.get("ckpt_dir", "./checkpoints/pairskate"),
                }
            }
        else:
            model_config = {
                "model_name": "pamfn_base",
                "model_conf": {
                    "in_dim": cur_conf["in_dim"]["V"],
                    "model_dim": cur_conf["model_dim"],
                    "drop_rate": cur_conf["drop_rate"],
                    "modality": args.modality
                }
            }
    elif not args.multi_modality:
        model_config = {
            "model_name": "pamfn_base",
            "model_conf": {
                "in_dim": cur_conf["in_dim"][args.modality],
                "model_dim": cur_conf["model_dim"],
                "drop_rate": cur_conf["drop_rate"],
                "modality": args.modality
            }
        }
    else:
        model_config = {
            "model_name": "pamfn_final",
            "model_conf": {
                "model_dim": cur_conf["model_dim"],
                "fc_drop": cur_conf["fc_drop"],
                "fc_r": cur_conf["fc_r"],
                "feat_drop": cur_conf["feat_drop"],
                "K": cur_conf["K"],
                "ms_heads": cur_conf["ms_heads"],
                "cm_heads": cur_conf["cm_heads"],
                "ckpt_dir": cur_conf["ckpt_dir"], 
                "rgb_ckpt_name": cur_conf["rgb_ckpt_name"], 
                "flow_ckpt_name": cur_conf["flow_ckpt_name"],
                "audio_ckpt_name": cur_conf["audio_ckpt_name"],
                "dataset_name": args.action
            }
        }

    optimizer_config = {
        "optim": cur_conf["optim"],
        "lr": cur_conf["lr"],
        "weight_decay": cur_conf["weight_decay"],
        "reduce_fc_lr": cur_conf.get("reduce_fc_lr", False)
    }

    lr_scheduler = {
        "scheduler": cur_conf["lr_scheduler"],
        "epoch_num": cur_conf["epoch"].get(args.action, cur_conf["epoch"]["TES"]),
        "scheduler_steps": cur_conf["epoch"],
        "lr_min": cur_conf["lr_min"],
        "warmup_lr_init": cur_conf["warmup_lr_init"],
        "decay_rate": 0.1
    }

    if args.print_flops:
        from pipeline import print_flops
        macs, params = print_flops(model_config)
        print(macs)
        print(params)
        exit()

    log_func = partial(log_info, logger=None, print_log=True)
    
    if args.test:
        if args.ckpt_path == "":
            if args.action == "PairSkate":
                ckpt_path = os.path.join(cur_conf["ckpt_dir"], args.action + "_multimodal.pth")
            else:
                ckpt_path = os.path.join(cur_conf["ckpt_dir"], args.action+"_multimodal.pth")
        else:
            ckpt_path = args.ckpt_path
            
        test_model(model_kwargs=model_config, dataset_kwargs=dataset_config,
                   ckpt_path=ckpt_path, log_func=log_func, seed=0, use_infer=True)
    else:
        ret = go(model_config, optimizer_config, lr_scheduler, dataset_config, cur_conf["epoch"].get(args.action, 250),
            save_base_model=args.save_base_model, save_moniter=cur_conf["save_moniter"], save_final_model=args.save_final_model,
            log_func=log_func, seed=0, clip_grad=cur_conf["clip_grad"], dataset_name=args.action)
        print("%.4f\t%.4f\t%.4f\t%.4f" % (ret[0], ret[1], ret[2], ret[3]))
