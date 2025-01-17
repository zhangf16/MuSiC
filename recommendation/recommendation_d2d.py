import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# import DiffModel
# import DiffModel_new as DiffModel
import DiffModel_short as DiffModel
import pandas as pd
import numpy as np
import h5py
import pickle
import gc
import os
import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class UserFeatureDataset(Dataset):
    def __init__(self, train_target_user, auxiliary_features, target_features):
        
        self.train_target_user = train_target_user
        self.auxiliary_features = auxiliary_features
        self.target_features = target_features

    def __len__(self):
        return len(self.target_features)
    
    def __getitem__(self, idx):
        user_id = self.train_target_user[idx]
        aux_feature_tensor = torch.tensor(self.auxiliary_features[idx], dtype=torch.float32)
        target_feature_tensor = torch.tensor(self.target_features[idx], dtype=torch.float32)
        return user_id, aux_feature_tensor, target_feature_tensor

class UserFeatureDataset2(Dataset):
    def __init__(self, train_target_user, target_features):
        
        self.train_target_user = train_target_user
        self.target_features = target_features

    def __len__(self):
        return len(self.target_features)
    
    def __getitem__(self, idx):
        user_id = self.train_target_user[idx]
        target_feature_tensor = torch.tensor(self.target_features[idx], dtype=torch.float32)
        return user_id, target_feature_tensor
    

def getDiffusionData():
    path1 = "/root/autodl-tmp/multimodal_diffusion/data/data1/data1/"
    with open(path1 + "train_source_user_feature.pkl", "rb") as f:
        train_source_user_feature = pickle.load(f)

    with open(path1 + "train_target_user_feature.pkl", "rb") as f:
        train_target_user_feature = pickle.load(f)

    train_source_user = list(train_source_user_feature.keys())
    train_target_user = list(train_target_user_feature.keys())
    other_user = [uid for uid in train_target_user if uid not in train_source_user]

    aux_features = [train_source_user_feature[uid] for uid in train_source_user]
    tgt_features = [train_source_user_feature[uid] for uid in train_source_user]

    other_features = [train_target_user_feature[uid] for uid in other_user]

    # 创建数据集和数据加载器
    dataset = UserFeatureDataset(train_target_user, aux_features, tgt_features)
    test_dataset = UserFeatureDataset2(other_user, tgt_features)
    return dataset, test_dataset

def create_updated_hdf5(predicted_features, original_hdf5_path, new_hdf5_path):

    with h5py.File(original_hdf5_path, 'r') as original_file, \
         h5py.File(new_hdf5_path, 'w') as new_file:

        # 读取原始数据
        user_ids = original_file['user_ids'][...]
        item_ids = original_file['item_ids'][...]
        original_user_features = original_file['user_features'][...]
        item_features = original_file['item_features'][...]
        ratings = original_file['ratings'][...]

        # 创建新的数据集
        new_user_features_ds = new_file.create_dataset('user_features', original_user_features.shape, dtype='float32')
        new_file.create_dataset('item_features', data=item_features)
        new_file.create_dataset('ratings', data=ratings)
        new_file.create_dataset('user_ids', data=user_ids)
        new_file.create_dataset('item_ids', data=item_ids)

        # 更新用户特征向量
        for i, user_id in enumerate(user_ids):
            if user_id in predicted_features:
                new_user_features_ds[i] = predicted_features[user_id]
            else:
                new_user_features_ds[i] = original_user_features[i]

# 定义数据集
class RatingsDataset1(Dataset):
    def __init__(self, hdf5_path):
        with h5py.File(hdf5_path, 'r') as file:
            self.user_id = file['user_ids'][...]
            self.user_features = file['user_features'][...]
            self.item_features = file['item_features'][...]
            self.ratings = file['ratings'][...]

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user_id = self.user_id[idx]
        user_features = self.user_features[idx]
        item_feature = self.item_features[idx]
        rating = self.ratings[idx]
        return torch.tensor(user_id), torch.tensor(user_features), torch.tensor(item_feature), torch.tensor(rating)

class RatingsDataset2(Dataset):
    def __init__(self, hdf5_path):
        with h5py.File(hdf5_path, 'r') as file:
            self.user_id = file['user_ids'][...]
            self.user_features_s = file['user_features_s'][...]
            self.user_features_t = file['user_features_t'][...]
            self.item_features = file['item_features'][...]
            self.ratings = file['ratings'][...]

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user_id = self.user_id[idx]
        user_features_s = self.user_features_s[idx]
        user_features_t = self.user_features_t[idx]
        item_feature = self.item_features[idx]
        rating = self.ratings[idx]
        return torch.tensor(user_id), torch.tensor(user_features_s), torch.tensor(user_features_t), torch.tensor(item_feature), torch.tensor(rating)

print('loading')
trainDiffusion, testDiffusion = getDiffusionData()
DiffusionData1 = DataLoader(trainDiffusion, batch_size=128, shuffle=True, num_workers=4)
DiffusionData2 = DataLoader(testDiffusion, batch_size=128, shuffle=True, num_workers=4)


path1 = "/root/autodl-tmp/multimodal_diffusion/data/data1/data1/"
with open(path1 + "train_source_user_feature.pkl", "rb") as f:
    train_source_user_feature = pickle.load(f)
with open(path1 + "train_target_user_feature.pkl", "rb") as f:
    train_target_user_feature = pickle.load(f)
with open(path1 + "test_target_user_feature.pkl", "rb") as f:
    test_target_user_feature = pickle.load(f)

num_users = len(set(train_target_user_feature.keys()) | set(test_target_user_feature.keys()))
train_hdf5_path1 = path1 + 'domain2_train_other_feature.hdf5'
train_dataset1 = RatingsDataset1(train_hdf5_path1)
train_hdf5_path2 = path1 + 'domain2_train_overlapping_feature.hdf5'
train_dataset2 = RatingsDataset2(train_hdf5_path2)
test_hdf5_path = path1 + "domain2_test_data_feature_new.hdf5"
test_dataset = RatingsDataset1(test_hdf5_path)
train_loader1 = DataLoader(train_dataset1, batch_size=256, shuffle=True)
train_loader2 = DataLoader(train_dataset2, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.best_loss = float('inf')
    
    def __call__(self, val_loss, model):
        score = -val_loss
        early_stop_info = ""

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                early_stop_info = f"EarlyStopping counter: {self.counter} out of {self.patience}"
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return early_stop_info
    
# 定义 MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, user_feature, item_feature):
        combined_feature = torch.cat((user_feature, item_feature), dim=1)
        return self.layers(combined_feature)


# 训练模型
def train_model(model, diff_model, train_loader1, train_loader2, test_loader, DiffusionData1, DiffusionData2, epochs, lr, weight_decay, non_overlap_weight):
    criterion = nn.MSELoss()
    optimizer_diff = torch.optim.Adam(params = diff_model.parameters(), lr= lr)
    optimizer = optim.Adam(list(model.parameters()) + list(diff_model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_diff, 'min', patience=5)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    early_stopping = EarlyStopping(patience=10, verbose=True)

    best_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        diff_model.train()

        total_loss = 0
        # predicted_features = {}
        # diffusion_losses = {}


        for user_id, aux_features, target_features in DiffusionData1:
            aux_features, target_features = aux_features.to(device), target_features.to(device)
            diffusion_loss = DiffModel.diffusion_loss(diff_model, target_features, device, aux_features)
            optimizer_diff.zero_grad()
            diffusion_loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(diff_model.parameters(),1.)
            optimizer_diff.step()

        for user_id, target_features in DiffusionData2:
            target_features = target_features.to(device)
            diffusion_loss = DiffModel.diffusion_loss(diff_model, target_features, device)
            diffusion_loss = diffusion_loss * non_overlap_weight
            optimizer_diff.zero_grad()
            diffusion_loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(diff_model.parameters(),1.)
            optimizer_diff.step()
        
        for user_id, user_feature, item_feature, rating in tqdm.tqdm(train_loader1, smoothing=0, mininterval=1.0):
        # for user_id, user_feature, item_feature, rating, is_overlap in train_loader:
            user_id, user_feature, item_feature, rating = user_id.to(device), user_feature.to(device), item_feature.to(device), rating.to(device)
            predicted_feature = DiffModel.p_sample(diff_model, user_feature, device)
            predicted_rating = model(predicted_feature.float(), item_feature.float()).sum(1)
            loss = criterion(predicted_rating, rating.float()) * non_overlap_weight
            optimizer.zero_grad()
            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(diff_model.parameters(),1.)
            optimizer.step()
        
        for user_id, user_feature_s, user_feature_t, item_feature, rating in tqdm.tqdm(train_loader2, smoothing=0, mininterval=1.0):
        # for user_id, user_feature, item_feature, rating, is_overlap in train_loader:
            user_id, user_feature_s, user_feature_t, item_feature, rating = user_id.to(device), user_feature_s.to(device), user_feature_t.to(device), item_feature.to(device), rating.to(device)
            predicted_feature = DiffModel.p_sample(diff_model, user_feature_s, device, user_feature_s)
            predicted_rating = model(predicted_feature.float(), item_feature.float()).sum(1)
            loss = criterion(predicted_rating, rating.float())
            optimizer.zero_grad()
            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(diff_model.parameters(),1.)
            optimizer.step()
        
        model.eval()
        diff_model.eval()

        actuals = []
        predictions = []
        with torch.no_grad():
            for user_id, x0, item_feature, rating in test_loader:
                user_id, x0, item_feature, rating = user_id.to(device), x0.to(device), item_feature.to(device), rating.to(device)
                
                predicted_feature = DiffModel.p_sample(diff_model, x0, device, x0)
                predicted_rating = model(predicted_feature.float(), item_feature.float()).sum(1)
                predictions.extend(predicted_rating.tolist())
                actuals.extend(rating.tolist())
        rmse, mae = calculate_rmse_mae(actuals, predictions)
        print(f"Epoch {epoch}, RMSE: {rmse:.5f}, MAE: {mae:.5f}", end='\t')

        # 检查是否是最佳模型
        if mae < best_loss:
            best_loss = mae
            best_epoch = epoch

        scheduler1.step(mae)
        scheduler2.step(mae)
        early_stop_info = early_stopping(mae, diff_model)
        print(early_stop_info)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    return best_loss, best_epoch

# 测试模型
def test_model(model, test_loader):
    model.eval()
    actuals = []
    predictions = []

    with torch.no_grad():
        for user_id, user_feature, item_feature, rating in test_loader:
            user_id, user_feature, item_feature, rating = user_id.to(device), user_feature.to(device), item_feature.to(device), rating.to(device)
            predicted_rating = model(user_feature.float(), item_feature.float()).sum(1)
            predictions.extend(predicted_rating.tolist())
            actuals.extend(rating.tolist())

    return actuals, predictions

# 计算 RMSE 和 MAE
def calculate_rmse_mae(actuals, predictions):
    mse = np.mean([(a - p) ** 2 for a, p in zip(actuals, predictions)])
    mae = np.mean([abs(a - p) for a, p in zip(actuals, predictions)])
    rmse = np.sqrt(mse)
    return rmse, mae

lr = 0.000001
epochs = 30

# hidden_dims = [768//2, 768//4]
# dropout = 0.2
# mask_rate = 0.1
# non_overlap_weight = 1.0
# T = 100
# t = 1

for T in [5,10,20]:
    for t in [0,1,2,4]:
        for mask_rate in [0.1,0.3,0.5]:
            for non_overlap_weight in [1.0,0.5]:
                for hidden_dims in [[768//2], [768//2, 768//4]]:
                    for dropout in [0.1,0.3]:
                        print(f'T: {T}, t: {t}, mask_rate: {mask_rate}, non_overlap_weight: {non_overlap_weight}')

                        # 初始化 MLP 模型
                        model = MLP(input_dim=768, hidden_dim=128, dropout_rate=dropout).to(device)
                        diff_model = DiffModel.DiffCDR(num_steps=1000, T=T,t=t,  in_features=768, diff_mask_rate=mask_rate,
                                                        hidden_dims=hidden_dims, dropout=dropout).to(device)
                        # 训练和测试
                        print('training')
                        trained_model = train_model(model, diff_model, train_loader1, train_loader2, test_loader, DiffusionData1, DiffusionData2, 
                                                    epochs=epochs, lr=lr, weight_decay=0.0001, non_overlap_weight = non_overlap_weight)
# print('testing')
# actuals, predictions = test_model(trained_model, test_loader)
# rmse, mae = calculate_rmse_mae(actuals, predictions)
# print(f"RMSE: {rmse}, MAE: {mae}")
