import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import h5py
import pickle
import gc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


# 定义 MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.layers(x)


# 定义数据集
class RatingsDataset(Dataset):
    def __init__(self, hdf5_path):
        with h5py.File(hdf5_path, 'r') as file:
            self.user_features = file['user_features'][...]
            self.item_features = file['item_features'][...]
            self.ratings = file['ratings'][...]

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user_feature = self.user_features[idx]
        item_feature = self.item_features[idx]
        rating = self.ratings[idx]
        return torch.tensor(user_feature), torch.tensor(item_feature), torch.tensor(rating)


# 训练模型
def train_model(model, train_loader, epochs, lr, weight_decay):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(model['user_mlp'].parameters()) + list(model['item_mlp'].parameters()), lr=lr, weight_decay=weight_decay)
    
    for epoch in range(epochs):
        model['user_mlp'].train()
        model['item_mlp'].train()
        total_loss = 0
        for user_feature, item_feature, rating in train_loader:
            user_feature, item_feature, rating = user_feature.to(device), item_feature.to(device), rating.to(device)
            user_output = model['user_mlp'](user_feature.float())
            item_output = model['item_mlp'](item_feature.float())
            predicted_rating = (user_output * item_output).sum(1)
            loss = criterion(predicted_rating, rating.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}, Loss: {average_loss}")
        
        model['user_mlp'].eval()
        model['item_mlp'].eval()

        actuals = []
        predictions = []
        with torch.no_grad():
            for user_feature, item_feature, rating in test_loader1:
                user_feature, item_feature, rating = user_feature.to(device), item_feature.to(device), rating.to(device)
                user_output = model['user_mlp'](user_feature.float())
                item_output = model['item_mlp'](item_feature.float())
                predicted_rating = (user_output * item_output).sum(1)
                predictions.extend(predicted_rating.tolist())
                actuals.extend(rating.tolist())
        rmse, mae = calculate_rmse_mae(actuals, predictions)
        print(f"RMSE: {rmse}, MAE: {mae}")

        actuals = []
        predictions = []
        with torch.no_grad():
            for user_feature, item_feature, rating in test_loader2:
                user_feature, item_feature, rating = user_feature.to(device), item_feature.to(device), rating.to(device)
                user_output = model['user_mlp'](user_feature.float())
                item_output = model['item_mlp'](item_feature.float())
                predicted_rating = (user_output * item_output).sum(1)
                predictions.extend(predicted_rating.tolist())
                actuals.extend(rating.tolist())
        rmse, mae = calculate_rmse_mae(actuals, predictions)
        print(f"RMSE: {rmse}, MAE: {mae}")

        

    return model

# 测试模型
def test_model(model, test_loader):
    model['user_mlp'].eval()
    model['item_mlp'].eval()
    actuals = []
    predictions = []

    with torch.no_grad():
        for user_feature, item_feature, rating in test_loader:
            user_feature, item_feature, rating = user_feature.to(device), item_feature.to(device), rating.to(device)
            user_output = model['user_mlp'](user_feature.float())
            item_output = model['item_mlp'](item_feature.float())
            predicted_rating = (user_output * item_output).sum(1)
            predictions.extend(predicted_rating.tolist())
            actuals.extend(rating.tolist())

    return actuals, predictions

# 计算 RMSE 和 MAE
def calculate_rmse_mae(actuals, predictions):
    mse = np.mean([(a - p) ** 2 for a, p in zip(actuals, predictions)])
    mae = np.mean([abs(a - p) for a, p in zip(actuals, predictions)])
    rmse = np.sqrt(mse)
    return rmse, mae


print('loading')
path1 = "/root/autodl-tmp/multimodal_diffusion/data/data1/data1/"
train_hdf5_path = path1 + 'domain2_all_train_data_feature.hdf5'
train_dataset = RatingsDataset(train_hdf5_path)


test_hdf5_path = path1 + "domain2_test_data_feature_new.hdf5"


test_dataset1 = RatingsDataset(test_hdf5_path)


test_diffuse_path = path1 + 'diffuse_user_feature.hdf5'
with open(path1 + "diffuse_user_feature.pkl", "rb") as f:
# with open(path1 + "test_target_user_feature.pkl", "rb") as f:
    predicted_features = pickle.load(f)
create_updated_hdf5(predicted_features, test_hdf5_path, test_diffuse_path)
test_dataset2 = RatingsDataset(test_diffuse_path)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader1 = DataLoader(test_dataset1, batch_size=64, shuffle=False)
test_loader2 = DataLoader(test_dataset2, batch_size=64, shuffle=False)

dropout_rate = 0.5
lr = 0.001

# 初始化 MLP 模型
user_mlp = MLP(input_dim=768, hidden_dim=128, dropout_rate=dropout_rate).to(device)
item_mlp = MLP(input_dim=768, hidden_dim=128).to(device)
model = {
    'user_mlp': user_mlp,
    'item_mlp': item_mlp
}

# 训练和测试
print('training')
trained_model = train_model(model, train_loader, epochs=10, lr=lr, weight_decay=0.0001)
# print('testing')
# actuals, predictions = test_model(trained_model, test_loader)
# rmse, mae = calculate_rmse_mae(actuals, predictions)
# print(f"RMSE: {rmse}, MAE: {mae}")

del user_mlp
del item_mlp
gc.collect()
torch.cuda.empty_cache()