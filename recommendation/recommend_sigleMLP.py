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
import mlp_model
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

class RatingsDataset2(Dataset):
    def __init__(self, hdf5_path):
        with h5py.File(hdf5_path, 'r') as file:
            self.user_features_s = file['user_features_s'][...]
            self.user_features_t = file['user_features_t'][...]
            self.item_features = file['item_features'][...]
            self.ratings = file['ratings'][...]

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user_features_t = self.user_features_t[idx]
        item_feature = self.item_features[idx]
        rating = self.ratings[idx]
        return torch.tensor(user_features_t), torch.tensor(item_feature), torch.tensor(rating)


# 训练模型
def train_model(model, train_loader1, train_loader2, epochs, lr, weight_decay):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    early_stopping = EarlyStopping(patience=10, verbose=True)

    best_rmse = float('inf')
    best_mae = float('inf')
    best_epoch = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for user_feature, item_feature, rating in train_loader1:
            user_feature, item_feature, rating = user_feature.to(device), item_feature.to(device), rating.to(device)
            predicted_rating = model(user_feature.float(), item_feature.float()).sum(1)
            loss = criterion(predicted_rating, rating.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        for user_feature, item_feature, rating in train_loader2:
            user_feature, item_feature, rating = user_feature.to(device), item_feature.to(device), rating.to(device)
            predicted_rating = model(user_feature.float(), item_feature.float()).sum(1)
            loss = criterion(predicted_rating, rating.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        average_loss = total_loss / (len(train_loader1) + len(train_loader1))
        print(f"Epoch {epoch}, Loss: {average_loss}")
        
        model.eval()

        actuals = []
        predictions = []
        with torch.no_grad():
            for user_feature, item_feature, rating in test_loader1:
                user_feature, item_feature, rating = user_feature.to(device), item_feature.to(device), rating.to(device)
                predicted_rating = model(user_feature.float(), item_feature.float()).sum(1)
                predictions.extend(predicted_rating.tolist())
                actuals.extend(rating.tolist())
        rmse, mae = calculate_rmse_mae(actuals, predictions)
        print(f"RMSE: {rmse}, MAE: {mae}")

        # actuals = []
        # predictions = []
        # with torch.no_grad():
        #     for user_feature, item_feature, rating in test_loader2:
        #         user_feature, item_feature, rating = user_feature.to(device), item_feature.to(device), rating.to(device)
        #         predicted_rating = model(user_feature.float(), item_feature.float()).sum(1)
        #         predictions.extend(predicted_rating.tolist())
        #         actuals.extend(rating.tolist())
        # rmse, mae = calculate_rmse_mae(actuals, predictions)
        # print(f"RMSE: {rmse}, MAE: {mae}")

        # 检查是否是最佳模型
        if rmse < best_rmse:
            best_rmse = rmse
            best_mae = mae
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
        
        scheduler.step(rmse)
        early_stop_info = early_stopping(rmse, model)
        print(early_stop_info)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return best_rmse,best_mae, best_epoch, best_model_state


# 计算 RMSE 和 MAE
def calculate_rmse_mae(actuals, predictions):
    mse = np.mean([(a - p) ** 2 for a, p in zip(actuals, predictions)])
    mae = np.mean([abs(a - p) for a, p in zip(actuals, predictions)])
    rmse = np.sqrt(mse)
    return rmse, mae


# path1 = "/root/autodl-tmp/multimodal_diffusion/data/book2movie/2/"
print('loading')
for path1 in [
    "/root/autodl-tmp/multimodal_diffusion/data/book2movie/2/",
    "/root/autodl-tmp/multimodal_diffusion/data/book2movie/5/",
    "/root/autodl-tmp/multimodal_diffusion/data/book2movie/8/",
    "/root/autodl-tmp/multimodal_diffusion/data/movie2music/2/",
    "/root/autodl-tmp/multimodal_diffusion/data/movie2music/5/",
    "/root/autodl-tmp/multimodal_diffusion/data/movie2music/8/",
    "/root/autodl-tmp/multimodal_diffusion/data/book2music/2/",
    "/root/autodl-tmp/multimodal_diffusion/data/book2music/5/",
    "/root/autodl-tmp/multimodal_diffusion/data/book2music/8/"
]:
    
    all_train_data_csv = pd.read_csv(path1 + 'domain2_all_train_data_copy.csv')
    item_popularity = all_train_data_csv['parent_asin'].value_counts().reset_index()
    item_popularity.columns = ['parent_asin', 'popularity']
    # print(item_popularity)

    train_hdf5_path1 = os.path.dirname(path1.rstrip('/')) + '/domain2_train_other_feature.hdf5'
    # train_hdf5_path = path1 + 'domain2_all_train_data_feature.hdf5'
    train_dataset1 = mlp_model.RatingsDataset(train_hdf5_path1)

    train_hdf5_path2 = path1 + 'domain2_train_overlapping_feature.hdf5'
    train_dataset2 = RatingsDataset2(train_hdf5_path2)

    test_hdf5_path = path1 + "domain2_test_data_feature_new.hdf5"
    test_dataset1 = mlp_model.RatingsDataset(test_hdf5_path)


    # test_diffuse_path = path1 + 'diffuse_user_feature.hdf5'
    # with open(path1 + "diffuse_user_feature.pkl", "rb") as f:
    # # with open(path1 + "test_target_user_feature.pkl", "rb") as f:
    #     predicted_features = pickle.load(f)
    # mlp_model.create_updated_hdf5(predicted_features, test_hdf5_path, test_diffuse_path)
    # test_dataset2 = mlp_model.RatingsDataset(test_diffuse_path)

    train_loader1 = DataLoader(train_dataset1, batch_size=512, shuffle=True)
    train_loader2 = DataLoader(train_dataset2, batch_size=512, shuffle=True)
    test_loader1 = DataLoader(test_dataset1, batch_size=512, shuffle=False)
    # test_loader2 = DataLoader(test_dataset2, batch_size=512, shuffle=False)

    # dropout_rate = 0.5
    lr = 0.00001

    # 初始化 MLP 模型
    for dropout_rate in [0.1]:
        model = mlp_model.MLP(input_dim=768, hidden_dim=128, dropout_rate=dropout_rate).to(device)
        # 训练和测试
        print('training')
        best_rmse, best_mae, best_epoch, best_model_state = train_model(model, train_loader1, train_loader2, epochs=40, lr=lr, weight_decay=0.0001)
        print(f'best_epoch {best_epoch}, best_rmse {best_rmse}, best_mae {best_mae}')
        # 保存模型状态字典到文件
        torch.save(best_model_state, path1 + 'simpleMLP_best_model_state.pth')




# print('testing')
# def test_model(model, test_loader):
#     model.eval()
#     actuals = []
#     predictions = []

#     results_df = pd.DataFrame(columns=['user_id', 'parent_asin', 'actual_rating', 'predicted_rating'])


#     with torch.no_grad():
#         for user_id, parent_asin, user_feature, item_feature, rating in test_loader:
#             user_id, parent_asin, user_feature, item_feature, rating = user_id.to(device), parent_asin.to(device), user_feature.to(device), item_feature.to(device), rating.to(device)
#             predicted_rating = model(user_feature.float(), item_feature.float()).sum(1)

#             batch_results = pd.DataFrame({
#                 'user_id': user_id.cpu().numpy(),
#                 'parent_asin': parent_asin.cpu().numpy(),
#                 'actual_rating': rating.cpu().numpy(),
#                 'predicted_rating': predicted_rating.cpu().numpy()
#             })
#             results_df = pd.concat([results_df, batch_results], ignore_index=True)

#             predictions.extend(predicted_rating.tolist())
#             actuals.extend(rating.tolist())
#     rmse, mae = calculate_rmse_mae(actuals, predictions)
#     print(f"RMSE: {rmse:.5f}, MAE: {mae:.5f}", end='\t')
#     return results_df

# test_dataset2 = mlp_model.RatingsDataset2(test_hdf5_path)
# test_loader2 = DataLoader(test_dataset2, batch_size=64, shuffle=False)
# model = mlp_model.MLP(input_dim=768, hidden_dim=128, dropout_rate=0.3).to(device)
# model_state = torch.load(path1 + 'simpleMLP_best_model_state.pth')
# model.load_state_dict(model_state)
# # model.load_state_dict(best_model_state)
# result = test_model(model, test_loader2)


# #ndcg
# def dcg_at_k(scores, k):
#     return np.sum([
#         (2 ** rel - 1) / np.log2(idx + 1) for idx, rel in enumerate(scores[:k], start=1)
#     ])

# def ndcg_at_k(predicted_scores, true_scores, k):
#     idcg = dcg_at_k(true_scores, k)
#     dcg = dcg_at_k(predicted_scores, k)
#     return dcg / idcg if idcg > 0 else 0

# all_ndcg_scores = []
# for user_id, group in result.groupby('user_id'):
#     actual_ratings = group.sort_values(by='actual_rating', ascending=False)['actual_rating'].tolist()
#     predicted_ratings = group.sort_values(by='predicted_rating', ascending=False)['actual_rating'].tolist()
#     k = 20
#     user_ndcg = ndcg_at_k(predicted_ratings, actual_ratings, k)
#     all_ndcg_scores.append(user_ndcg)

# average_ndcg = np.mean(all_ndcg_scores)
# print("Average NDCG:", average_ndcg)


# #新颖性
# def calculate_novelty(recommendations, item_popularity):
#     novelty_scores = []
#     for parent_asin in recommendations:
#         popularity = item_popularity.loc[item_popularity['parent_asin'] == parent_asin, 'popularity'].values[0]
#         novelty_scores.append(1 / popularity if popularity > 0 else 0)
#     return np.mean(novelty_scores)

# all_novelty_scores = []
# for user_id, group in result.groupby('user_id'):
#     recommended_items = group.sort_values(by='predicted_rating', ascending=False)['parent_asin'].tolist()[:10]
#     user_novelty_score = calculate_novelty(recommended_items, item_popularity)
#     all_novelty_scores.append(user_novelty_score)

# average_novelty = np.mean(all_novelty_scores)
# print("Average Novelty:", average_novelty)



# # 1. 识别长尾物品
# tail_items = item_popularity[item_popularity['popularity'] < 10]['parent_asin']

# # 2. 筛选预测结果
# tail_results = result[result['parent_asin'].isin(tail_items)]

# # 3. 计算 MAE 和 RMSE
# tail_rmse, tail_mae = calculate_rmse_mae(tail_results['actual_rating'], tail_results['predicted_rating'])

# print(f"Long Tail RMSE: {tail_rmse:.5f}, MAE: {tail_mae:.5f}")
