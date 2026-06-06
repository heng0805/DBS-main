import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange


class BaseConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.GELU()
        )

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class PairSkateBaseModel(nn.Module):
    def __init__(self, in_dim, model_dim, drop_rate):
        super().__init__()
        dim = model_dim
        
        self.rgb_embedding = nn.Sequential(
            nn.Conv1d(in_dim["V"], 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.Dropout(0.3)
        )
        
        self.flow_embedding = nn.Sequential(
            nn.Conv1d(in_dim["F"], 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.Dropout(0.3)
        )
        
        self.audio_embedding = nn.Sequential(
            nn.Conv1d(in_dim["A"], 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.Dropout(0.3)
        )
        
        self.seq1_embedding = nn.Sequential(
            nn.Conv1d(in_dim["S1"], 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.Dropout(0.3)
        )
        
        self.seq2_embedding = nn.Sequential(
            nn.Conv1d(in_dim["S2"], 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.Dropout(0.3)
        )
        
        self.stage1 = BaseConvBlock(dim)
        self.stage2 = BaseConvBlock(dim)
        self.stage3 = BaseConvBlock(dim)
        self.pool = nn.AvgPool1d(2, 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Conv1d(dim * 5, dim, 1),
            nn.ReLU(True),
            nn.Dropout(drop_rate),
            nn.Conv1d(dim, 1, 1),
            nn.Sigmoid()
        )
        self.mse = nn.MSELoss()

    def forward(self, feats):
        rgb = feats["V"]
        flow = feats["F"]
        audio = feats["A"]
        seq1 = feats["S1"]
        seq2 = feats["S2"]
        
        if len(rgb.shape) == 4:
            rgb = rgb.mean(dim=2)
            flow = flow.mean(dim=2)
            audio = audio.mean(dim=2)
            seq1 = seq1.mean(dim=2)
            seq2 = seq2.mean(dim=2)
        
        rgb = rearrange(rgb, 'b t d -> b d t').contiguous()
        flow = rearrange(flow, 'b t d -> b d t').contiguous()
        audio = rearrange(audio, 'b t d -> b d t').contiguous()
        seq1 = rearrange(seq1, 'b t d -> b d t').contiguous()
        seq2 = rearrange(seq2, 'b t d -> b d t').contiguous()
        
        rgb = self.rgb_embedding(rgb)
        flow = self.flow_embedding(flow)
        audio = self.audio_embedding(audio)
        seq1 = self.seq1_embedding(seq1)
        seq2 = self.seq2_embedding(seq2)
        
        x1 = self.stage1(rgb + flow + audio + seq1 + seq2)
        x1 = self.pool(x1)

        x2 = self.stage2(x1)
        x2 = self.pool(x2)

        x3 = self.stage3(x2)
        x3 = self.gap(x3)

        x_combined = torch.cat([rgb[:, :, :1], flow[:, :, :1], audio[:, :, :1], seq1[:, :, :1], seq2[:, :, :1]], dim=1)
        x_combined = x_combined.repeat(1, 1, x3.shape[-1])
        
        score = self.fc(x3).squeeze(dim=2)
        return score, {'feats': [x1, x2, x3]}

    def call_loss(self, pred, label, **kwargs):
        return self.mse(pred.squeeze(), label.squeeze())


class Multi_Head_Attention(nn.Module):
    def __init__(self, num_heads, dim_model, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        assert dim_model % num_heads == 0
        self.dim_head = dim_model // num_heads
        self.linear_layers = nn.ModuleList([nn.Linear(dim_model, dim_model) for _ in range(3)])
        self.drop = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, input_Q, input_K, input_V, mask=None):
        b = input_V.size(0)
        Q, K, V = [l(x).view(b, -1, self.num_heads, self.dim_head).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (input_Q, input_K, input_V))]
        
        scores = -1 * torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(Q.size(-1))
        if mask is not None:
            mask = mask.unsqueeze(dim=1).repeat([1, self.num_heads, 1, 1])
            scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        if self.drop is not None:
            attn = self.drop(attn)
        context = torch.matmul(attn, V)
        
        x = context.transpose(1, 2).contiguous().view(b, -1, self.num_heads * self.dim_head)
        return x


class LFC(nn.Module):
    """
    Leader-Follower Coordination Module
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.cross_attn = Multi_Head_Attention(num_heads=num_heads, dim_model=dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, leader_feat, follower_feat):
        b, t, d = leader_feat.shape
        leader_feat_norm = self.norm(leader_feat)
        follower_feat_norm = self.norm(follower_feat)
        
        attended = self.cross_attn(leader_feat_norm, follower_feat_norm, follower_feat_norm)
        output = leader_feat + attended
        return output


class RCF(nn.Module):
    """
    Rhythm-Aware Cross-Modal Fusion Module
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.cross_attn = Multi_Head_Attention(num_heads=num_heads, dim_model=dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, leader_feat, follower_feat, rhythm):
        leader_feat_norm = self.norm(leader_feat)
        follower_feat_norm = self.norm(follower_feat)
        rhythm_norm = self.norm(rhythm)
        
        attended = self.cross_attn(rhythm_norm, leader_feat_norm, follower_feat_norm)
        output = follower_feat + attended
        return output


class DBSModel(nn.Module):
    """
    Dual-Branch Synergistic (DBS) Model for Pair Skating Assessment
    """
    def __init__(self,
                 model_dim, fc_drop, fc_r, feat_drop, K,
                 ms_heads, cm_heads, lfc_heads,
                 in_dim,
                 use_lfc=True, use_rcf=True,
                 lambda_coord=0.1, lambda_rhythm=0.1,
                 ckpt_dir=None, dataset_name=None,
                 **kwargs):
        super().__init__()
        
        self.use_lfc = use_lfc
        self.use_rcf = use_rcf
        self.lambda_coord = lambda_coord
        self.lambda_rhythm = lambda_rhythm
        
        self.rgb_embedding = nn.Sequential(
            nn.Conv1d(in_dim["V"], 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, model_dim, 1),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(True),
            nn.Dropout(0.3)
        )
        
        self.flow_embedding = nn.Sequential(
            nn.Conv1d(in_dim["F"], 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, model_dim, 1),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(True),
            nn.Dropout(0.3)
        )
        
        self.audio_embedding = nn.Sequential(
            nn.Conv1d(in_dim["A"], 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, model_dim, 1),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(True),
            nn.Dropout(0.3)
        )
        
        self.seq1_embedding = nn.Sequential(
            nn.Conv1d(in_dim["S1"], 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, model_dim, 1),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(True),
            nn.Dropout(0.3)
        )
        
        self.seq2_embedding = nn.Sequential(
            nn.Conv1d(in_dim["S2"], 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, model_dim, 1),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(True),
            nn.Dropout(0.3)
        )
        
        if use_lfc:
            self.lfc = LFC(model_dim, lfc_heads)
            self.rcf = RCF(model_dim, cm_heads)
        
        self.stage1 = BaseConvBlock(model_dim)
        self.stage2 = BaseConvBlock(model_dim)
        self.stage3 = BaseConvBlock(model_dim)
        
        self.pool = nn.AvgPool1d(2, 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(feat_drop)
        
        self.fc = nn.Sequential(
            nn.Dropout(fc_drop),
            nn.Conv1d(model_dim, model_dim // fc_r, 1),
            nn.BatchNorm1d(model_dim // fc_r),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(model_dim // fc_r, 1, 1),
            nn.Sigmoid()
        )
        
        self.fc_tes = nn.Sequential(
            nn.Dropout(fc_drop),
            nn.Conv1d(model_dim, model_dim // fc_r, 1),
            nn.BatchNorm1d(model_dim // fc_r),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(model_dim // fc_r, 1, 1),
            nn.Sigmoid()
        )
        
        self.fc_pcs = nn.Sequential(
            nn.Dropout(fc_drop),
            nn.Conv1d(model_dim, model_dim // fc_r, 1),
            nn.BatchNorm1d(model_dim // fc_r),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(model_dim // fc_r, 1, 1),
            nn.Sigmoid()
        )
        
        self.mse = nn.MSELoss()

    def forward(self, feats):
        rgb = feats["V"]
        flow = feats["F"]
        audio = feats["A"]
        seq1 = feats["S1"]
        seq2 = feats["S2"]
        
        if len(rgb.shape) == 4:
            rgb = rgb.mean(dim=2)
            flow = flow.mean(dim=2)
            audio = audio.mean(dim=2)
            seq1 = seq1.mean(dim=2)
            seq2 = seq2.mean(dim=2)
        
        rgb = rearrange(rgb, 'b t d -> b d t').contiguous()
        flow = rearrange(flow, 'b t d -> b d t').contiguous()
        audio = rearrange(audio, 'b t d -> b d t').contiguous()
        seq1 = rearrange(seq1, 'b t d -> b d t').contiguous()
        seq2 = rearrange(seq2, 'b t d -> b d t').contiguous()
        
        rgb_emb = self.rgb_embedding(rgb)
        flow_emb = self.flow_embedding(flow)
        audio_emb = self.audio_embedding(audio)
        seq1_emb = self.seq1_embedding(seq1)
        seq2_emb = self.seq2_embedding(seq2)
        
        coord_loss = 0.0
        rhythm_loss = 0.0
        
        if self.use_lfc:
            seq1_t = seq1_emb.permute(0, 2, 1)
            seq2_t = seq2_emb.permute(0, 2, 1)
            
            leader_feat = self.lfc(seq1_t, seq2_t)
            follower_feat = self.lfc(seq2_t, seq1_t)
            
            coord_diff = torch.abs(leader_feat - follower_feat).mean()
            coord_loss = self.lambda_coord * coord_diff
            
            leader_feat = leader_feat.permute(0, 2, 1)
            follower_feat = follower_feat.permute(0, 2, 1)
            
            combined_emb = rgb_emb + flow_emb + audio_emb + leader_feat + follower_feat
        else:
            combined_emb = rgb_emb + flow_emb + audio_emb + seq1_emb + seq2_emb
        
        x1 = self.stage1(combined_emb)
        x1 = self.pool(x1)

        x2 = self.stage2(x1)
        x2 = self.pool(x2)

        x3 = self.stage3(x2)
        x3_gap = self.gap(x3)
        
        score = self.fc(x3_gap).squeeze(dim=2)
        
        other_info = {
            'feats': [x1, x2, x3],
            'coord_loss': coord_loss,
            'rhythm_loss': rhythm_loss
        }
        
        return score, other_info

    def call_loss(self, pred, label, **kwargs):
        mse_loss = self.mse(pred.squeeze(), label.squeeze())
        coord_loss = kwargs.get('coord_loss', 0.0)
        rhythm_loss = kwargs.get('rhythm_loss', 0.0)
        
        total_loss = mse_loss + coord_loss + rhythm_loss
        return total_loss


PairSkateModel = DBSModel
