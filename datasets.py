import os
import os.path as osp
import torch
import numpy as np
from torch.utils.data import Dataset

def zero_padding_torch(data, target_length):
    original_length = data.shape[0]
    zero_data = torch.zeros((target_length-original_length, ) + data.shape[1:])
    return torch.cat([data, zero_data], dim=0)


def get_default_segments(dataset_names):
    if isinstance(dataset_names, list):
        dataset_name = dataset_names[0]
    else:
        dataset_name = dataset_names
    
    if dataset_name in ['Ball', 'Clubs', 'Hoop', 'Ribbon']:
        return 70
    elif dataset_name in ['TES', 'PCS', 'PairSkate']:
        return 130
    else:
        raise ValueError(f"数据集 {dataset_name} 不支持！")

def load_features_from_directory(directory_path):
    """从指定目录加载所有 .npy 特征文件"""
    features_dict = {}
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"目录 {directory_path} 不存在")
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory_path, filename)
            key = os.path.splitext(filename)[0]
            features_dict[key] = np.load(file_path)
    return features_dict


class PairSkateDataset(Dataset):
    def __init__(self, dataset_name='PairSkate', feat_dir=None,
             rgb_feat='VST', flow_feat='I3D', audio_feat='AST',
             is_train=True, segments=-1, **kwargs):
        super().__init__()
        self.is_train = is_train
        self.dataset_name = dataset_name
        if feat_dir is None:
            self.feat_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'datasets')
        else:
            self.feat_dir = feat_dir
        
        if segments < 0:
            self.segments = get_default_segments(dataset_name)
        else:
            self.segments = segments
            
        self.labels = self.build(is_train)
        self.datas = self.read_data(feat_dir, rgb_feat, flow_feat, audio_feat)
        
        self.validate_data_matching()

    def build(self, is_train):
        labels = []
        
        label_path = osp.join(self.feat_dir, "train.txt" if is_train else "test.txt")
        
        TES_MAX = 45.0
        PCS_MAX = 45.0
        TOTAL_MAX = 100.0
        
        with open(label_path, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.strip()
                if not line or line.startswith('ID') or line.startswith('Rank'):
                    continue
                
                parts = line.split()
                if len(parts) >= 4:
                    video_id = parts[0]
                    total_score = float(parts[1]) / TOTAL_MAX
                    pcs_score = float(parts[2]) / PCS_MAX
                    tes_score = float(parts[3]) / TES_MAX
                    td = float(parts[4]) if len(parts) > 4 else 0.0
                    
                    labels.append({
                        'video_id': video_id,
                        'total_score': total_score,
                        'pcs_score': pcs_score,
                        'tes_score': tes_score,
                        'td': td
                    })
        
        if not labels:
            raise ValueError(f"未找到标签数据！请检查 {label_path}")
        
        return labels

    def read_data(self, feat_dir, rgb_feat, flow_feat, audio_feat):
        if feat_dir is None:
            feat_dir = self.feat_dir
            
        rgb_dir = osp.join(feat_dir, "rgb_VST")
        flow_dir = osp.join(feat_dir, "flow_I3D")
        audio_dir = osp.join(feat_dir, "audio_AST")
        seq1_dir = osp.join(feat_dir, "skater1_feature")
        seq2_dir = osp.join(feat_dir, "skater2_feature")
        
        rgb_data = load_features_from_directory(rgb_dir) if os.path.exists(rgb_dir) else {}
        flow_data = load_features_from_directory(flow_dir) if os.path.exists(flow_dir) else {}
        audio_data = load_features_from_directory(audio_dir) if os.path.exists(audio_dir) else {}
        seq1_data = load_features_from_directory(seq1_dir) if os.path.exists(seq1_dir) else {}
        seq2_data = load_features_from_directory(seq2_dir) if os.path.exists(seq2_dir) else {}
        
        return {
            "rgb": rgb_data, 
            "flow": flow_data, 
            "audio": audio_data,
            "seq1": seq1_data,
            "seq2": seq2_data
        }

    def validate_data_matching(self):
        valid_labels = []
        missing_samples = []
        
        for item in self.labels:
            sample_name = item['video_id']
            if (sample_name in self.datas['rgb'] and 
                sample_name in self.datas['flow'] and 
                sample_name in self.datas['audio']):
                valid_labels.append(item)
            else:
                missing_samples.append(sample_name)
        
        if missing_samples:
            pass
        
        self.labels = valid_labels

    def __getitem__(self, idx):
        item = self.labels[idx]
        sample_name = item['video_id']
        total_score = item['total_score']
        pcs_score = item['pcs_score']
        tes_score = item['tes_score']
        td = item['td']
        
        rgb_data = torch.from_numpy(self.datas['rgb'][sample_name]).float()
        flow_data = torch.from_numpy(self.datas['flow'][sample_name]).float()
        audio_data = torch.from_numpy(self.datas['audio'][sample_name]).float()
        
        if sample_name in self.datas['seq1']:
            seq1_data = torch.from_numpy(self.datas['seq1'][sample_name]).float()
        else:
            seq1_data = torch.zeros_like(rgb_data)
            
        if sample_name in self.datas['seq2']:
            seq2_data = torch.from_numpy(self.datas['seq2'][sample_name]).float()
        else:
            seq2_data = torch.zeros_like(rgb_data)

        min_len = min(len(rgb_data), len(flow_data), len(audio_data), len(seq1_data), len(seq2_data))
        rgb_data = rgb_data[:min_len]
        flow_data = flow_data[:min_len]
        audio_data = audio_data[:min_len]
        seq1_data = seq1_data[:min_len]
        seq2_data = seq2_data[:min_len]

        if len(rgb_data) > self.segments:
            start_idx = 0 if not self.is_train else np.random.randint(len(rgb_data) - self.segments)
            rgb_data = rgb_data[start_idx:start_idx + self.segments]
            audio_data = audio_data[start_idx:start_idx + self.segments]
            flow_data = flow_data[start_idx:start_idx + self.segments]
            seq1_data = seq1_data[start_idx:start_idx + self.segments]
            seq2_data = seq2_data[start_idx:start_idx + self.segments]
        elif len(rgb_data) < self.segments:
            rgb_data = zero_padding_torch(rgb_data, self.segments)
            audio_data = zero_padding_torch(audio_data, self.segments)
            flow_data = zero_padding_torch(flow_data, self.segments)
            seq1_data = zero_padding_torch(seq1_data, self.segments)
            seq2_data = zero_padding_torch(seq2_data, self.segments)
        
        return {
            'rgb': rgb_data,
            'flow': flow_data,
            'audio': audio_data,
            'seq1': seq1_data,
            'seq2': seq2_data,
            'total_score': torch.tensor(total_score, dtype=torch.float32),
            'pcs_score': torch.tensor(pcs_score, dtype=torch.float32),
            'tes_score': torch.tensor(tes_score, dtype=torch.float32),
            'td': torch.tensor(td, dtype=torch.float32),
            'video_id': sample_name
        }

    def __len__(self):
        return len(self.labels)


class AQADataset(Dataset):
    def __init__(self, dataset_name=None, feat_dir=None,
             rgb_feat=None, flow_feat=None, audio_feat=None,
             squeeze_rgb_feat=None, squeeze_flow_feat=None,
             is_train=True, segments=-1, main_classes=None, auxiliary_classes=None, **kwargs):
        super().__init__()
        self.is_train = is_train
        self.squeeze_rgb_feat = squeeze_rgb_feat
        self.squeeze_flow_feat = squeeze_flow_feat
    
        if auxiliary_classes is not None:
            self.auxiliary_classes = auxiliary_classes if isinstance(auxiliary_classes, list) else [auxiliary_classes]
        elif 'auxiliary_classes' in kwargs and kwargs['auxiliary_classes'] is not None:
            self.auxiliary_classes = kwargs['auxiliary_classes'] if isinstance(kwargs['auxiliary_classes'], list) else [kwargs['auxiliary_classes']]
        else:
            self.auxiliary_classes = []
    
        if main_classes is not None:
            self.main_classes = main_classes if isinstance(main_classes, list) else [main_classes]
        elif dataset_name is not None:
            self.main_classes = dataset_name if isinstance(dataset_name, list) else [dataset_name]
        else:
            raise ValueError("必须提供 main_classes 或 dataset_name 参数")
    
        if self.is_train and self.auxiliary_classes:
            self.all_classes = self.main_classes + self.auxiliary_classes
        else:
            self.all_classes = self.main_classes
        
        if 'PairSkate' in self.all_classes or dataset_name == 'PairSkate':
            self.is_pair_skate = True
            self.pair_skate_dataset = PairSkateDataset(
                dataset_name='PairSkate',
                feat_dir=None,
                is_train=is_train,
                segments=segments
            )
            self.labels = self.pair_skate_dataset.labels
            self.datas = self.pair_skate_dataset.datas
            self.segments = self.pair_skate_dataset.segments
            return
        else:
            self.is_pair_skate = False
        
        self.labels = self.build(self.all_classes, is_train)
        self.datas = self.read_data(self.all_classes, feat_dir, rgb_feat, flow_feat, audio_feat)
        
        if segments < 0:
            self.segments = get_default_segments(self.all_classes)
        else:
            self.segments = segments
            
        print(f"[数据加载] 总样本数: {len(self.labels)}")
        print(f"[数据加载] RGB特征数: {len(self.datas['rgb'])}")
        print(f"[数据加载] Flow特征数: {len(self.datas['flow'])}")
        print(f"[数据加载] Audio特征数: {len(self.datas['audio'])}")
        
        self.validate_data_matching()

    def build(self, dataset_names, is_train):
        labels = []
        dataset_names = [dataset_names] if not isinstance(dataset_names, list) else dataset_names
        
        for dataset_name in dataset_names:
            if dataset_name not in ['Ball', 'Clubs', 'Hoop', 'Ribbon', 'TES', 'PCS']:
                raise ValueError(f"类别 {dataset_name} 不支持！")
            
            if dataset_name in ['Ball', 'Clubs', 'Hoop', 'Ribbon']:
                label_path = osp.join("./data/RG/", "train.txt" if is_train else "test.txt")
                with open(label_path, 'r') as fr:
                    lines = fr.readlines()
                    for i, line in enumerate(lines[1:]):
                        line = line.strip().split()
                        if line[0].startswith(dataset_name):
                            difficulty_score = float(line[1])
                            execution_score = float(line[2])
                            total_score = float(line[3]) / 25
                            group_label = int(line[6]) if len(line) > 6 else 0
                            labels.append((line[0], total_score, difficulty_score, execution_score, group_label))
            
            elif dataset_name in ["TES", "PCS"]:
                label_path = osp.join("./data/FISV/", "train.txt" if is_train else "test.txt")
                with open(label_path, 'r') as fr:
                    for line in fr.readlines():
                        line = line.strip().split()
                        if dataset_name == 'TES':
                            labels.append((line[0], float(line[1]) / 45))
                        elif dataset_name == 'PCS':
                            labels.append((line[0], float(line[2]) / 45))
        
        if not labels:
            raise ValueError(f"未找到类别 {dataset_names} 的标签！")
        
        class_counts = {}
        for item in labels:
            name = item[0]
            class_name = name.split('_')[0]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        print(f"[数据加载] 各类别样本数: {class_counts}")
            
        return labels

    def read_data(self, dataset_names, feat_dir, rgb_feat, flow_feat, audio_feat):
        rgb_data = {}
        flow_data = {}
        audio_data = {}
    
        for dataset_name in dataset_names:
            dir_name = "FISV" if dataset_name in ["TES", "PCS"] else dataset_name
        
            new_feat_base = "./data/features/"
            rgb_dir = osp.join(new_feat_base, f"{dir_name}_rgb_{rgb_feat}")
            flow_dir = osp.join(new_feat_base, f"{dir_name}_flow_{flow_feat}")
            audio_dir = osp.join(new_feat_base, f"{dir_name}_audio_{audio_feat}")

            try:
                cur_rgb = load_features_from_directory(rgb_dir)
                cur_flow = load_features_from_directory(flow_dir)
                cur_audio = load_features_from_directory(audio_dir)

            except FileNotFoundError:
                rgb_path = osp.join(new_feat_base, f"{dir_name}_rgb_{rgb_feat}.npy")
                flow_path = osp.join(new_feat_base, f"{dir_name}_flow_{flow_feat}.npy")
                audio_path = osp.join(new_feat_base, f"{dir_name}_audio_{audio_feat}.npy")
            
                if os.path.exists(rgb_path):
                    cur_rgb = np.load(rgb_path, allow_pickle=True).item()
                else:
                    cur_rgb = {}
                    
                if os.path.exists(flow_path):
                    cur_flow = np.load(flow_path, allow_pickle=True).item()
                else:
                    cur_flow = {}
                    
                if os.path.exists(audio_path):
                    cur_audio = np.load(audio_path, allow_pickle=True).item()
                else:
                    cur_audio = {}

            rgb_data.update(cur_rgb)
            flow_data.update(cur_flow)
            audio_data.update(cur_audio)

        class_feature_counts = {}
        for key in rgb_data:
            class_name = key.split('_')[0]
            class_feature_counts[class_name] = class_feature_counts.get(class_name, 0) + 1
    
        return {"rgb": rgb_data, "flow": flow_data, "audio": audio_data}

    def validate_data_matching(self):
        missing_samples = []
        for item in self.labels:
            sample_name = item[0] if isinstance(item, tuple) else item['video_id']
            if sample_name not in self.datas['rgb']:
                missing_samples.append(sample_name)

    def squeeze_dim(self, rgb_data, flow_data, audio_data):
        if self.squeeze_rgb_feat == 'mean' and len(rgb_data.shape) == 3:
            rgb_data = rgb_data.mean(dim=1)
        elif self.squeeze_rgb_feat == 'cat' and len(rgb_data.shape) == 3:
            rgb_data = rgb_data.view(rgb_data.shape[0], -1)
        if self.squeeze_flow_feat and len(flow_data.shape) == 3:
            flow_data = flow_data.mean(dim=1)
        elif self.squeeze_flow_feat and len(flow_data) == 3:
            flow_data = flow_data.view(flow_data.shape[0], -1)
        return rgb_data, flow_data, audio_data

    def __getitem__(self, idx):
        if self.is_pair_skate:
            return self.pair_skate_dataset[idx]
        
        item = self.labels[idx]
        sample_name = item[0]
        label = item[1]
        difficulty_label = item[2]
        execution_label = item[3]
        group_label = item[4] 
        group_label -=1

        class_name = sample_name.split('_')[0]

        class_to_idx = {'Clubs': 0, 'Ribbon': 1, 'Hoop': 2, 'Ball': 3}
        class_idx = class_to_idx.get(class_name, -1)
        if class_idx == -1:
            raise ValueError(f"未知类别：{class_name},请检查class_to_idx映射")
        
        rgb_data = torch.from_numpy(self.datas['rgb'][sample_name]).float()
        flow_data = torch.from_numpy(self.datas['flow'][sample_name]).float()
        audio_data = torch.from_numpy(self.datas['audio'][sample_name]).float()

        rgb_data, flow_data, audio_data = self.squeeze_dim(rgb_data, flow_data, audio_data)

        if self.is_train:
            if len(rgb_data) > self.segments:
                start_idx = np.random.randint(len(rgb_data) - self.segments)
                rgb_data = rgb_data[start_idx:start_idx + self.segments]
                audio_data = audio_data[start_idx:start_idx + self.segments]
                flow_data = flow_data[start_idx:start_idx + self.segments]
            elif len(rgb_data) < self.segments:
                rgb_data = zero_padding_torch(rgb_data, self.segments)
                audio_data = zero_padding_torch(audio_data, self.segments)
                flow_data = zero_padding_torch(flow_data, self.segments)
        
        is_main = 1 if class_name in self.main_classes else 0
        
        return (rgb_data, audio_data, flow_data, 
                label, difficulty_label, execution_label,
                is_main, group_label, class_idx)

    def __len__(self):
        return len(self.labels)
