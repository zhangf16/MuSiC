import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# import DiffModel
# import DiffModel_noise as DiffModel
import DiffModel_ldm as DiffModel
import pandas as pd
import numpy as np
import h5py
import pickle
import gc
import os
import tqdm
import copy
import time
import mlp_model
import json

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
    

def getDiffusionData(path1):
    with open(path1 + "train_source_user_feature.pkl", "rb") as f:
        train_source_user_feature = pickle.load(f)

    with open(path1 + "train_target_user_feature.pkl", "rb") as f:
        train_target_user_feature = pickle.load(f)

    train_source_user = list(train_source_user_feature.keys())
    train_target_user = list(train_target_user_feature.keys())
    other_user = [uid for uid in train_target_user if uid not in train_source_user]

    aux_features = [train_source_user_feature[uid] for uid in train_source_user]
    tgt_features = [train_target_user_feature[uid] for uid in train_source_user]

    other_features = [train_target_user_feature[uid] for uid in other_user]

    # 创建数据集和数据加载器
    dataset = UserFeatureDataset(train_source_user, aux_features, tgt_features)
    test_dataset = UserFeatureDataset2(other_user, other_features)
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

class RatingsDataset3(Dataset):
    def __init__(self, hdf5_path):
        with h5py.File(hdf5_path, 'r') as file:
            self.user_id = file['user_ids'][...]
            self.item_id = file['item_ids'][...]
            self.user_features = file['user_features'][...]
            self.item_features = file['item_features'][...]
            self.ratings = file['ratings'][...]

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user_id = self.user_id[idx]
        item_id = self.item_id[idx]
        user_features = self.user_features[idx]
        item_feature = self.item_features[idx]
        rating = self.ratings[idx]
        return torch.tensor(user_id), torch.tensor(item_id), torch.tensor(user_features), torch.tensor(item_feature), torch.tensor(rating)
    


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
    


class UserEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(UserEmbedding, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        # self.tanh = nn.Sigmoid()
        # self.tanh = nn.ReLU()
        # self.tanh = nn.Softmax(dim=1)
        self.dropout1 = nn.Dropout(dropout_rate)  # 添加Dropout层
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)  # 添加Dropout层
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.tanh(self.layer1(x))
        x = self.dropout1(x)  # 应用Dropout
        x = self.tanh(self.layer2(x))
        x = self.dropout2(x)  # 应用Dropout
        x = self.output_layer(x)
        return x

class ItemEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(ItemEmbedding, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        # self.tanh = nn.Sigmoid()
        # self.tanh = nn.ReLU()
        # self.tanh = nn.Softmax(dim=1)
        self.dropout1 = nn.Dropout(dropout_rate)  # 添加Dropout层
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)  # 添加Dropout层
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.tanh(self.layer1(x))
        x = self.dropout1(x)  # 应用Dropout
        x = self.tanh(self.layer2(x))
        x = self.dropout2(x)  # 应用Dropout
        x = self.output_layer(x)
        return x
    
# 训练模型
def train_model(UserEmbed, ItemEmbed, diff_model, train_loader1, train_loader2, test_loader, epochs, lr, weight_decay, non_overlap_weight, lamda):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(UserEmbed.parameters()) + list(ItemEmbed.parameters()) + list(diff_model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    early_stopping = EarlyStopping(patience=10, verbose=True)

    best_rmse = float('inf')
    best_mae = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        start_time = time.time()  # 开始时间
        UserEmbed.train()
        ItemEmbed.train()
        diff_model.train()


        for user_id, user_feature, item_feature, rating in train_loader1:
        # for user_id, user_feature, item_feature, rating, is_overlap in train_loader:
            user_id, user_feature, item_feature, rating = user_id.to(device), user_feature.to(device), item_feature.to(device), rating.to(device)
            user = UserEmbed(user_feature)
            item = ItemEmbed(item_feature)
            #with condition 
            # diffusion_loss = DiffModel.diffusion_loss(diff_model, user, device, user).mean()
            # predicted_feature = DiffModel.p_sample(diff_model, user, device, user)

            #no condition 
            diffusion_loss = DiffModel.diffusion_loss(diff_model, user, device).mean()
            # predicted_feature = DiffModel.p_sample(diff_model, user, device)
            # predicted_feature = DiffModel.p_sample_random(diff_model, user, device)
            # predicted_rating = (predicted_feature * item).sum(1)
            # loss = criterion(predicted_rating, rating.float())
            # total_loss = (lamda * diffusion_loss + (1 - lamda) * loss) * non_overlap_weight
            total_loss = diffusion_loss * non_overlap_weight
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        for user_id, user_feature_s, user_feature_t, item_feature, rating in train_loader2:
        # for user_id, user_feature, item_feature, rating, is_overlap in train_loader:
            user_id, user_feature_s, user_feature_t, item_feature, rating = \
                user_id.to(device), user_feature_s.to(device), user_feature_t.to(device), item_feature.to(device), rating.to(device)
            source = UserEmbed(user_feature_s)
            target = UserEmbed(user_feature_t)
            item = ItemEmbed(item_feature)
            diffusion_loss = DiffModel.diffusion_loss(diff_model, target, device, source).mean()
            # predicted_feature = DiffModel.p_sample(diff_model, source, device, source)
            predicted_feature = DiffModel.p_sample_random(diff_model, source, device, source)
            predicted_rating = (predicted_feature * item).sum(1)

            # # 1. 将 predicted_rating 转换为概率分布，用于排序
            # pred_probs = nn.functional.softmax(predicted_rating, dim=0)

            # # 2. 构建目标分布，评分越高的物品标签越高
            # sorted_ratings, indices = torch.sort(rating, descending=True)
            # true_probs = torch.zeros_like(pred_probs)
            # true_probs[indices] = nn.functional.softmax(sorted_ratings.float(), dim=0)

            # # 3. Listwise Softmax Cross-Entropy Loss
            # listwise_ranking_loss = nn.functional.kl_div(pred_probs.log(), true_probs, reduction='batchmean')

            # beta = 0.1

            loss = criterion(predicted_rating, rating.float())
            total_loss = lamda * diffusion_loss + (1 - lamda) * loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        UserEmbed.eval()
        ItemEmbed.eval()
        diff_model.eval()

        actuals = []
        predictions = []
        with torch.no_grad():
            for user_id, parent_asin, user_feature, item_feature, rating in test_loader:
                user_id, parent_asin, user_feature, item_feature, rating = user_id.to(device), parent_asin.to(device), user_feature.to(device), item_feature.to(device), rating.to(device)
                user = UserEmbed(user_feature)
                item = ItemEmbed(item_feature)
                # predicted_feature = DiffModel.p_sample(diff_model, user, device, user)
                predicted_feature = DiffModel.p_sample_random(diff_model, user, device, user)
                predicted_rating = (predicted_feature * item).sum(1)
                predictions.extend(predicted_rating.tolist())
                actuals.extend(rating.tolist())
        rmse, mae = mlp_model.calculate_rmse_mae(actuals, predictions)
        print(f"Epoch {epoch}, RMSE: {rmse:.5f}, MAE: {mae:.5f}", end='\t')

        # 检查是否是最佳模型
        if rmse < best_rmse:
            best_rmse = rmse
            best_epoch = epoch
            best_model_state1 = copy.deepcopy(UserEmbed.state_dict())
            best_model_state2 = copy.deepcopy(ItemEmbed.state_dict())
            best_model_state3 = copy.deepcopy(diff_model.state_dict())
        if mae < best_mae:
            best_mae = mae


        scheduler.step(rmse)
        early_stop_info = early_stopping(rmse, diff_model)
        

        end_time = time.time()  # 结束时间
        epoch_duration = end_time - start_time
        print(f"{epoch_duration:.2f} seconds", end='\t')
        print(early_stop_info)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    return best_mae, best_rmse, best_epoch, best_model_state1, best_model_state2, best_model_state3

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
            # predicted_feature = DiffModel.p_sample(diff_model, user, device, user)
            predicted_feature = DiffModel.p_sample_random(diff_model, user, device, user)
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
    rmse, mae = mlp_model.calculate_rmse_mae(actuals, predictions)
    # print(f"RMSE: {rmse:.5f}, MAE: {mae:.5f}", end='\t')
    return results_df, rmse, mae



def save_model_and_results(lr, T, t, dropout, non_overlap_weight, lamda, hidden_dims, attention, best_epoch, 
                           UserEmbed, ItemEmbed, diff_model, results):
    # 保存模型参数
    save_dir = path1 + "model_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_save_path = os.path.join(save_dir, f"model_lr{lr}_T{T}_t{t}_dropout{dropout}_overlap{non_overlap_weight}_lamda{lamda}_hidden{hidden_dims}_attention{attention}.pt")
    torch.save({
        "UserEmbed_state_dict": UserEmbed.state_dict(),
        "ItemEmbed_state_dict": ItemEmbed.state_dict(),
        "diff_model_state_dict": diff_model.state_dict(),
        "best_epoch": best_epoch
    }, model_save_path)
    
    # 保存结果
    results_save_path = os.path.join(save_dir, "results.json")
    with open(results_save_path, "a") as f:
        json.dump(results, f)
        f.write("\n")


lr = 0.0001
epochs = 100

# hidden_dims = [32*2]
hidden_dims = [32*2, 32*4]
# attention = [0,1]
attention = [0]
# attention = []
T = 10
t = 0
dropout = 0.2

non_overlap_weight = 1.0
lamda = 0.4


all_configurations = []
# for hidden_dims in [[32*2],[32*2, 32*4],[32*2, 32*4, 32*8]]:
# for attention in [[], [0],[0,1]]:
# for T in [5,10,50,100]:
# for t in [0,T//4,T//2,T]:
# for dropout in [0.2,0.4]:
# for non_overlap_weight in [0,0.2,0.4,0.6,0.8,1.0]:
# for lamda in [0.2,0.4,0.6,0.8,1.0]:


# for domain in ["movie2music/","book2movie/","book2music/"]:
for domain in ["book2music/"]:
    # for hidden_dims in [[32*2, 32*4, 32*8]]:
    for hidden_dims in [[32*2]]:
            for ratio in ["8/"]:
                # for non_overlap_weight in [1.0]:
                    # for lamda in [0.4]:
                        print('loading')
                        # ratio = "8/"
                        # path1 = "/root/autodl-tmp/multimodal_diffusion/data/movie2music/5/"
                        # path1 = "/root/autodl-tmp/multimodal_diffusion/data/book2movie/2/"
                        # path1 = "/root/autodl-tmp/multimodal_diffusion/data/book2music/8/"

                        path1 = "/root/autodl-tmp/multimodal_diffusion/data/"+domain+ratio

                        with open(path1 + "train_source_user_feature.pkl", "rb") as f:
                            train_source_user_feature = pickle.load(f)
                        with open(path1 + "train_target_user_feature.pkl", "rb") as f:
                            train_target_user_feature = pickle.load(f)
                        with open(path1 + "test_target_user_feature.pkl", "rb") as f:
                            test_target_user_feature = pickle.load(f)

                        num_users = len(set(train_target_user_feature.keys()) | set(test_target_user_feature.keys()))
                        train_hdf5_path1 = os.path.dirname(path1.rstrip('/')) + '/domain2_train_other_feature.hdf5'
                        train_dataset1 = RatingsDataset1(train_hdf5_path1)
                        train_hdf5_path2 = path1 + 'domain2_train_overlapping_feature.hdf5'
                        train_dataset2 = RatingsDataset2(train_hdf5_path2)
                        test_hdf5_path = path1 + "domain2_test_data_feature_new.hdf5"
                        test_new_hdf5_path = path1 + "domain2_test_data_feature_new.hdf5"
                        test_none_hdf5_path = path1 + "domain2_test_data_feature_none.hdf5"
                        train_loader1 = DataLoader(train_dataset1, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)
                        train_loader2 = DataLoader(train_dataset2, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)
                        test_dataset = RatingsDataset3(test_hdf5_path)
                        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
                        test_new_dataset = RatingsDataset3(test_new_hdf5_path)
                        test_new_loader = DataLoader(test_new_dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
                        test_none_dataset = RatingsDataset3(test_none_hdf5_path)
                        test_none_loader = DataLoader(test_none_dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)

                        all_train_data_csv = pd.read_csv(path1 + 'domain2_all_train_data_copy.csv')
                        item_popularity = all_train_data_csv['parent_asin'].value_counts().reset_index()
                        item_popularity.columns = ['parent_asin', 'popularity']


                        print(f'path: {path1}, lr: {lr}, T: {T}, dropout: {dropout}, lamda: {lamda}, hidden_dims: {hidden_dims}')
                        UserEmbed = UserEmbedding(input_size=4096, hidden_size=128, output_size=32, dropout_rate=dropout).to(device)
                        ItemEmbed = ItemEmbedding(input_size=4096, hidden_size=128, output_size=32, dropout_rate=dropout).to(device)
                        diff_model = DiffModel.DiffCDR(num_steps=T, t=t,  in_features=32, hidden_dims=hidden_dims, dropout=dropout, attn_resolutions=attention).to(device)
                        # 训练和测试
                        print('training')
                        best_mae, best_rmse, best_epoch, best_model_state1, best_model_state2, best_model_state3 = \
                            train_model(UserEmbed, ItemEmbed, diff_model, train_loader1, train_loader2, test_new_loader, \
                            epochs=epochs, lr=lr, weight_decay=0.0001, non_overlap_weight = non_overlap_weight, lamda=lamda)
                        
                        UserEmbed.load_state_dict(best_model_state1)
                        ItemEmbed.load_state_dict(best_model_state2)
                        diff_model.load_state_dict(best_model_state3)
                        
                        result, rmse_new, mae_new = test_model(UserEmbed, ItemEmbed, diff_model, test_new_loader)
                        ndcg = mlp_model.ndcg(result)
                        novel = mlp_model.novelty(result,item_popularity)
                        tail_rmse, tail_mae = mlp_model.longtail(result,item_popularity)
                        result, rmse, mae = test_model(UserEmbed, ItemEmbed, diff_model, test_loader)
                        result, rmse_none, mae_none = test_model(UserEmbed, ItemEmbed, diff_model, test_none_loader)

                        
                        config_result = {
                                "lr": lr,
                                "T": T,
                                "t": t,
                                "dropout": dropout,
                                "non_overlap_weight": non_overlap_weight,
                                "lamda": lamda,
                                "hidden_dims": hidden_dims,
                                "attention": attention,
                                "best_mae_new": best_mae,
                                "best_rmse_new": best_rmse,
                                "best_epoch": best_epoch,
                                "mae": mae,
                                "rmse": rmse,
                                "mae_none": mae_none,
                                "rmse_none": rmse_none,
                                "ndcg": ndcg,
                                "novel": novel,
                                "tail_rmse": tail_rmse,
                                "tail_mae": tail_mae
                            }
                        print(config_result)
                        # 保存模型参数和结果
                        # save_model_and_results(lr, T, t, dropout, non_overlap_weight, lamda, hidden_dims, attention, best_epoch, UserEmbed, ItemEmbed, diff_model, config_result)

                        all_configurations.append({
                                "lr": lr,
                                "T": T,
                                "t": t,
                                "dropout": dropout,
                                "non_overlap_weight": non_overlap_weight,
                                "lamda": lamda,
                                "hidden_dims": hidden_dims,
                                "attention": attention,
                                "best_mae_new": best_mae,
                                "best_rmse_new": best_rmse,
                                "best_epoch": best_epoch,
                                "mae": mae,
                                "rmse": rmse,
                                "mae_none": mae_none,
                                "rmse_none": rmse_none,
                                "ndcg": ndcg,
                                "novel": novel,
                                "tail_rmse": tail_rmse,
                                "tail_mae": tail_mae
                            })
print()
for config in all_configurations:
    print(config)
