import os
import sys
import argparse
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
import configs.PairSkate_config as conf
from datasets import PairSkateDataset
from models.dbs_pairskate import DBSModel


def setup_seed(seed):
    import numpy as np
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  


def test_pairskate(gpu='0', ckpt_path=None, score_type='total_score'):
    if gpu != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    setup_seed(0)
    
    cur_conf = conf.pair_skate_v9
    
    SCORE_MAX = {
        'tes_score': 45.0,
        'pcs_score': 45.0,
        'total_score': 100.0
    }
    
    print(f"=" * 60)
    print(f"测试配置:")
    print(f"  - 评分类型: {score_type}")
    print(f"  - GPU: {gpu}")
    print(f"  - 模型路径: {ckpt_path}")
    print(f"=" * 60)
    
    test_dset = PairSkateDataset(is_train=False, segments=cur_conf.get("segments", 130))
    test_dloader = DataLoader(test_dset, batch_size=1, shuffle=False,
                              pin_memory=False, num_workers=4)
    
    model = DBSModel(
        model_dim=cur_conf["model_dim"],
        fc_drop=cur_conf.get("fc_drop", 0),
        fc_r=cur_conf.get("fc_r", 2),
        feat_drop=cur_conf.get("feat_drop", 0.5),
        K=cur_conf.get("K", 6),
        ms_heads=cur_conf.get("ms_heads", 1),
        cm_heads=cur_conf.get("cm_heads", 1),
        lfc_heads=cur_conf.get("lfc_heads", 1),
        in_dim=cur_conf.get("in_dim", {"V": 1024, "F": 1024, "A": 768, "S1": 1024, "S2": 1024}),
        use_lfc=cur_conf.get("use_lfc", True),
        use_rcf=cur_conf.get("use_rcf", True),
        lambda_coord=cur_conf.get("lambda_coord", 0.1),
        lambda_rhythm=cur_conf.get("lambda_rhythm", 0.1),
    )
    
    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        print(f"已加载模型: {ckpt_path}")
    else:
        print(f"警告: 模型文件不存在 {ckpt_path}")
        return
    
    model.cuda()
    model.eval()
    
    total_preds = []
    total_labels = []
    total_loss = 0.0
    mse = torch.nn.MSELoss()
    
    print("\n开始测试...")
    with torch.no_grad():
        for i, batch in enumerate(test_dloader):
            rgb_data = batch['rgb'].cuda()
            flow_data = batch['flow'].cuda()
            audio_data = batch['audio'].cuda()
            seq1_data = batch['seq1'].cuda()
            seq2_data = batch['seq2'].cuda()
            total_score = batch['total_score'].float().cuda()
            pcs_score = batch['pcs_score'].float().cuda()
            tes_score = batch['tes_score'].float().cuda()
            video_id = batch['video_id'][0]
            
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
            loss = mse(preds.squeeze(), label.squeeze())
            total_loss += loss.item()
            
            pred_val = preds.squeeze().cpu().detach().numpy().tolist()
            if isinstance(pred_val, float):
                total_preds.append(pred_val)
            else:
                total_preds.extend(pred_val)
            
            total_labels.extend(label.cpu().detach().numpy().tolist())
            
            if (i + 1) % 20 == 0:
                print(f"已处理 {i + 1}/{len(test_dloader)} 样本")
    
    avg_loss = total_loss / len(test_dloader)
    coef, p_value = spearmanr(total_preds, total_labels)
    
    score_max = SCORE_MAX.get(score_type, 100.0)
    original_preds = np.array(total_preds) * score_max
    original_labels = np.array(total_labels) * score_max
    
    rmse = np.sqrt(np.mean((original_preds - original_labels) ** 2))
    mean_label = np.mean(np.abs(original_labels))
    relative_l2 = rmse / mean_label if mean_label > 0 else 0.0
    
    print(f"\n" + "=" * 60)
    print(f"测试结果:")
    print(f"  - Spearman相关系数 (ρ): {coef:.4f}")
    print(f"  - P值: {p_value:.6f}")
    print(f"  - R-L2: {relative_l2:.4f}")
    print(f"  - 测试样本数: {len(total_labels)}")
    print(f"=" * 60)
    
    print(f"\n归一化预测值范围: [{min(total_preds):.4f}, {max(total_preds):.4f}]")
    print(f"归一化真实值范围: [{min(total_labels):.4f}, {max(total_labels):.4f}]")
    print(f"\n原始预测值范围: [{min(original_preds):.2f}, {max(original_preds):.2f}]")
    print(f"原始真实值范围: [{min(original_labels):.2f}, {max(original_labels):.2f}]")
    
    return {
        'spearman_coef': coef,
        'p_value': p_value,
        'relative_l2': relative_l2,
        'avg_loss': avg_loss,
        'predictions': total_preds,
        'labels': total_labels
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PairSkate Model')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--score_type', type=str, default='total_score', 
                        choices=['total_score', 'tes_score', 'pcs_score'],
                        help='Score type to evaluate')
    
    args = parser.parse_args()
    
    test_pairskate(
        gpu=args.gpu,
        ckpt_path=args.ckpt_path,
        score_type=args.score_type
    )
