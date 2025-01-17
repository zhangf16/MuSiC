import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import h5py
import pickle
import gc
import copy
# import DiffModel_short as DiffModel
import DiffModel_noise as DiffModel

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

class predictDataset(Dataset):
    def __init__(self, predicted_features, original_hdf5_path):
        with h5py.File(original_hdf5_path, 'r') as original_file:
            # 读取原始数据
            user_ids = original_file['user_ids'][...]
            item_features = original_file['item_features'][...]
            ratings = original_file['ratings'][...]
            new_user_features_ds = []
            for i, user_id in enumerate(user_ids):
                new_user_features_ds.append(predicted_features[user_id])
        self.user_features = new_user_features_ds
        self.item_features = item_features
        self.ratings = ratings
    
    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user_feature = self.user_features[idx]
        item_feature = self.item_features[idx]
        rating = self.ratings[idx]
        return torch.tensor(user_feature), torch.tensor(item_feature), torch.tensor(rating)

        

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


class RatingsDataset2(Dataset):
    def __init__(self, hdf5_path):
        with h5py.File(hdf5_path, 'r') as file:
            self.user_id = file['user_ids'][...]
            self.parent_asin = file['item_ids'][...]
            self.user_features = file['user_features'][...]
            self.item_features = file['item_features'][...]
            self.ratings = file['ratings'][...]

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user_id = self.user_id[idx]
        parent_asin = self.parent_asin[idx]
        user_feature = self.user_features[idx]
        item_feature = self.item_features[idx]
        rating = self.ratings[idx]
        return torch.tensor(user_id), torch.tensor(parent_asin), torch.tensor(user_feature), torch.tensor(item_feature), torch.tensor(rating)

def test_model(UserEmbed, ItemEmbed, diff_model, test_loader):
    UserEmbed.eval()
    ItemEmbed.eval()
    diff_model.eval()
    actuals = []
    predictions = []

    results_df = pd.DataFrame(columns=['user_id', 'parent_asin', 'actual_rating', 'predicted_rating'])


    with torch.no_grad():
        for user_id, parent_asin, user_feature, item_feature, rating in test_loader:
            user_id, parent_asin, user_feature, item_feature, rating = user_id.to(device), parent_asin.to(device), user_feature.to(device), item_feature.to(device), rating.to(device)
            # predicted_rating = model(user_feature.float(), item_feature.float()).sum(1)
            user = UserEmbed(user_feature)
            item = ItemEmbed(item_feature)
            predicted_feature = DiffModel.p_sample(diff_model, user, device, user)
            predicted_rating = (predicted_feature * item).sum(1)

            batch_results = pd.DataFrame({
                'user_id': user_id.cpu().numpy(),
                'parent_asin': parent_asin.cpu().numpy(),
                'actual_rating': rating.cpu().numpy(),
                'predicted_rating': predicted_rating.cpu().numpy()
            })
            results_df = pd.concat([results_df, batch_results], ignore_index=True)

            predictions.extend(predicted_rating.tolist())
            actuals.extend(rating.tolist())
    rmse, mae = calculate_rmse_mae(actuals, predictions)
    # print(f"RMSE: {rmse:.5f}, MAE: {mae:.5f}", end='\t')
    return results_df, rmse, mae

# 计算 RMSE 和 MAE
def calculate_rmse_mae(actuals, predictions):
    mse = np.mean([(a - p) ** 2 for a, p in zip(actuals, predictions)])
    mae = np.mean([abs(a - p) for a, p in zip(actuals, predictions)])
    rmse = np.sqrt(mse)
    return rmse, mae

def dcg_at_k(scores, k):
    return np.sum([
        (2 ** rel - 1) / np.log2(idx + 1) for idx, rel in enumerate(scores[:k], start=1)
    ])

def ndcg_at_k(predicted_scores, true_scores, k):
    idcg = dcg_at_k(true_scores, k)
    dcg = dcg_at_k(predicted_scores, k)
    return dcg / idcg if idcg > 0 else 0

def ndcg(result):
    all_ndcg_scores = []
    for user_id, group in result.groupby('user_id'):
        actual_ratings = group.sort_values(by='actual_rating', ascending=False)['actual_rating'].tolist()
        predicted_ratings = group.sort_values(by='predicted_rating', ascending=False)['actual_rating'].tolist()
        k = 20
        user_ndcg = ndcg_at_k(predicted_ratings, actual_ratings, k)
        all_ndcg_scores.append(user_ndcg)

    average_ndcg = np.mean(all_ndcg_scores)
    return average_ndcg
    # print("Average NDCG:", average_ndcg)

def calculate_novelty(recommendations, item_popularity):
    novelty_scores = []
    for parent_asin in recommendations:
        popularity = item_popularity.loc[item_popularity['parent_asin'] == parent_asin, 'popularity'].values[0]
        novelty_scores.append(1 / popularity if popularity > 0 else 0)
    return np.mean(novelty_scores)

def novelty(result,item_popularity):
    all_novelty_scores = []
    for user_id, group in result.groupby('user_id'):
        recommended_items = group.sort_values(by='predicted_rating', ascending=False)['parent_asin'].tolist()[:10]
        user_novelty_score = calculate_novelty(recommended_items, item_popularity)
        all_novelty_scores.append(user_novelty_score)

    average_novelty = np.mean(all_novelty_scores)
    # print("Average Novelty:", average_novelty)
    return average_novelty

def longtail(result,item_popularity):
    # 1. 识别长尾物品
    tail_items = item_popularity[item_popularity['popularity'] < 10]['parent_asin']

    # 2. 筛选预测结果
    tail_results = result[result['parent_asin'].isin(tail_items)]

    # 3. 计算 MAE 和 RMSE
    tail_rmse, tail_mae = calculate_rmse_mae(tail_results['actual_rating'], tail_results['predicted_rating'])

    # print(f"Long Tail RMSE: {tail_rmse:.5f}, MAE: {tail_mae:.5f}")
    return tail_rmse, tail_mae