
import pandas as pd
import numpy as np
import h5py
import random
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import tqdm
import mlp_model
import copy
import recommend_diffusion_ablation as diff
import DiffModel_short as DiffModel
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 假设的文件路径（请替换为实际的文件路径）

ratio = 8

file_path1 = "/root/autodl-tmp/data/movie/data.csv"
file_path2 = "/root/autodl-tmp/data/music/data.csv"
path2 = '/root/code/DiffCDR-main/data/ready/_' + str(10-ratio)+ '_' + str(ratio)+ '/tgt_CDs_and_Vinyl_src_Movies_and_TV/'

# file_path1 = "/root/autodl-tmp/data/book/data.csv"
# file_path2 = "/root/autodl-tmp/data/movie/data.csv"
# path2 = '/root/code/DiffCDR-main/data/ready/_' + str(10-ratio)+ '_' + str(ratio)+ '/tgt_Movies_and_TV_src_Books/'

# file_path1 = "/root/autodl-tmp/data/book/data.csv"
# file_path2 = "/root/autodl-tmp/data/music/data.csv"
# path2 = '/root/code/DiffCDR-main/data/ready/_' + str(10-ratio)+ '_' + str(ratio)+ '/tgt_CDs_and_Vinyl_src_Books/'


# 加载两个域的数据
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

src = df1
tgt = df2
src_users = set(src.user_id.unique())
tgt_users = set(tgt.user_id.unique())
co_users = src_users & tgt_users

overlap_df1 = df1[df1['user_id'].isin(co_users)]
overlap_df2 = df2[df2['user_id'].isin(co_users)]
other_df2 = df2[~df2['user_id'].isin(co_users)]

co_users = sorted(list(co_users))
random.seed(2024)  # 设置固定的随机种子
test_users = random.sample(co_users, round(0.5 * len(co_users)))

train_df2 = overlap_df2[~overlap_df2['user_id'].isin(test_users)]
all_train_data2 = pd.concat([train_df2, other_df2])
test_df2 = overlap_df2[overlap_df2['user_id'].isin(test_users)]
test_df2_new = test_df2[test_df2['parent_asin'].isin(all_train_data2['parent_asin'].unique())]

# def get_combined_encoding(df1, df2, id_column):
#     # Combine and get unique IDs from both dataframes
#     combined_ids = pd.concat([df1[id_column], df2[id_column]]).unique()

#     # Map to new IDs
#     id_map = {id: i for i, id in enumerate(combined_ids)}

#     return id_map

# # Get combined user and item encodings
# user_id_map = get_combined_encoding(df1, df2, 'user_id')
# item_id_map = get_combined_encoding(df1, df2, 'parent_asin')

def load_dicts(file_path):
    with open(file_path, 'rb') as f:
        dicts = pickle.load(f)
    return dicts['uid_dict'], dicts['iid_dict_src'], dicts['iid_dict_tgt']




# 调用函数加载字典
uid_dict, iid_dict_src, iid_dict_tgt = load_dicts(path2 + 'dict')


# Function to apply combined encoding to a dataframe
def apply_combined_encoding(dataframe, user_id_map, item_id_map):
    dataframe['user_id'] = dataframe['user_id'].map(user_id_map)
    dataframe['parent_asin'] = dataframe['parent_asin'].map(item_id_map)
    return dataframe

# Apply the combined encoding to both dataframes
all_df1 = apply_combined_encoding(df1.copy(), uid_dict, iid_dict_src)
all_train_df2 = apply_combined_encoding(all_train_data2.copy(), uid_dict, iid_dict_tgt)
overlap_df2 = apply_combined_encoding(train_df2.copy(), uid_dict, iid_dict_tgt)
test_df2_new = apply_combined_encoding(test_df2_new.copy(), uid_dict, iid_dict_tgt)
other_df2 = apply_combined_encoding(other_df2.copy(), uid_dict, iid_dict_tgt)

# Dataset class for PyTorch
class RatingDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        user_id = int(row['user_id'])
        item_id = int(row['parent_asin'])
        rating = row['rating']  # Assuming a 'rating' column exists
        return torch.tensor([user_id, item_id], dtype=torch.long), torch.tensor(rating, dtype=torch.float32)


# Assuming the maximum IDs and embedding dimensions
max_user_id = max(uid_dict.values())+1
max_item_id = max(iid_dict_tgt.values())+1
embedding_dim = 32  # Example embedding dimension

model = diff.DNNBasedModel(max_user_id, max_item_id, embedding_dim, 50)
model.load_state_dict(torch.load(path2 + 'model.pth'))

dataset3 = RatingDataset(test_df2_new)
dataloader3 = DataLoader(dataset3, batch_size=10240, shuffle=True)

#test
# predicted_scores = []
# actural_scores = []
# all_loss = 0
# criterion = nn.MSELoss()

# # 遍历测试数据
# for inputs, ratings in dataloader3:
#     # inputs, ratings = inputs.to(device), ratings.to(device)
#     uid_emb = model.tgt_model.uid_embedding(inputs[:, 0].unsqueeze(1))
#     iid_emb = model.tgt_model.iid_embedding(inputs[:, 1].unsqueeze(1))
#     emb = torch.cat([uid_emb, iid_emb], dim=1)
#     scores = torch.sum(model.tgt_model.linear(emb[:, 0, :]) * emb[:, 1, :], 1)
    
#     # user_ids = inputs[:, 0]
#     # item_ids = inputs[:, 1]
#     # src_emb = model.tgt_model.linear(model.tgt_model.uid_embedding(user_ids.unsqueeze(1)))
#     # iid_emb = model.tgt_model.iid_embedding(item_ids.unsqueeze(1))
#     # emb = torch.cat([src_emb, iid_emb], 1)
#     # scores = torch.sum(emb[:, 0, :] * emb[:, 1, :], 1)
#     loss = criterion(scores, ratings)
#     all_loss+=loss.item()
#     predicted_scores.append(scores.cpu().detach().numpy())
#     actural_scores.append(ratings.cpu().detach().numpy())

# print(all_loss / len(dataloader3))
# predicted_scores = np.concatenate(predicted_scores)
# actural_scores = np.concatenate(actural_scores)
# rmse, mae = diff.calculate_rmse_mae(actural_scores, predicted_scores)
# print(f"RMSE: {rmse}, MAE: {mae}")


aux_domain_user_embeddings = model.src_model.linear(model.src_model.uid_embedding.weight.data).detach().cpu().numpy()
target_domain_user_embeddings = model.tgt_model.linear(model.tgt_model.uid_embedding.weight.data).detach().cpu().numpy()

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
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
    
class TargetDomainDataset1(Dataset):
    def __init__(self, dataframe, user_features, item_features):
        self.dataframe = dataframe
        self.user_features = user_features
        self.item_features = item_features

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        user_id = int(row['user_id'])
        item_id = int(row['parent_asin'])
        rating = row['rating']
        user_feature = self.user_features[user_id]
        item_feature = self.item_features[item_id]
        return user_id, torch.tensor(user_feature), torch.tensor(item_feature), torch.tensor(rating, dtype=torch.float32)

class TargetDomainDataset2(Dataset):
    def __init__(self, dataframe, user_features, item_features):
        self.dataframe = dataframe
        self.user_features = user_features
        self.item_features = item_features

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        user_id = int(row['user_id'])
        item_id = int(row['parent_asin'])
        rating = row['rating']
        user_feature = self.user_features[user_id]
        item_feature = self.item_features[item_id]
        return user_id, item_id, torch.tensor(user_feature), torch.tensor(item_feature), torch.tensor(rating, dtype=torch.float32)

class OverlapUsersDataset1(Dataset):
    def __init__(self, dataframe, user_features_source, user_features_target, item_features):
        self.dataframe = dataframe
        self.user_features_source = user_features_source
        self.user_features_target = user_features_target
        self.item_features = item_features

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        user_id = int(row['user_id'])
        item_id = int(row['parent_asin'])
        rating = row['rating']
        user_feature_s = self.user_features_source[user_id]
        user_feature_t = self.user_features_target[user_id]
        item_feature = self.item_features[item_id]
        return user_id, torch.tensor(user_feature_s), torch.tensor(user_feature_t), torch.tensor(item_feature), torch.tensor(rating, dtype=torch.float32)


class TargetDomainOnlyDataset(Dataset):
    def __init__(self, embeddings, user_ids):
        self.embeddings = embeddings
        self.user_ids = user_ids

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        return self.embeddings[user_id]

class OverlapUsersDataset(Dataset):
    def __init__(self, aux_embeddings, target_embeddings, user_ids):
        self.aux_embeddings = aux_embeddings
        self.target_embeddings = target_embeddings
        self.user_ids = user_ids

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        return self.aux_embeddings[user_id], self.target_embeddings[user_id]

class TestDataset(Dataset):
    def __init__(self, aux_embeddings, target_embeddings, user_ids):
        self.aux_embeddings = aux_embeddings
        self.target_embeddings = target_embeddings
        self.user_ids = user_ids

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        return self.aux_embeddings[user_id], self.target_embeddings[user_id], user_id

class UserEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(UserEmbedding, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout_rate)  # 添加Dropout层
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)  # 添加Dropout层
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.tanh(self.layer1(x))
        x = self.dropout1(x)  # 应用Dropout
        # x = self.tanh(self.layer2(x))
        # x = self.dropout2(x)  # 应用Dropout
        x = self.output_layer(x)
        return x

class ItemEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(ItemEmbedding, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout_rate)  # 添加Dropout层
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)  # 添加Dropout层
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.tanh(self.layer1(x))
        x = self.dropout1(x)  # 应用Dropout
        # x = self.tanh(self.layer2(x))
        # x = self.dropout2(x)  # 应用Dropout
        x = self.output_layer(x)
        return x

# 用户ID集合
target_only_user_ids = list(set(other_df2.user_id.unique()))
overlap_user_ids = list(set(overlap_df2.user_id.unique()))
test_user_ids = list(set(test_df2_new.user_id.unique()))

# 创建DataLoader
target_only_dataset = TargetDomainOnlyDataset(target_domain_user_embeddings, target_only_user_ids)
target_only_dataloader = DataLoader(target_only_dataset, batch_size=64, shuffle=True)

overlap_dataset = OverlapUsersDataset(aux_domain_user_embeddings, target_domain_user_embeddings, overlap_user_ids)
overlap_dataloader = DataLoader(overlap_dataset, batch_size=64, shuffle=True)

test_dataset = TestDataset(aux_domain_user_embeddings, target_domain_user_embeddings, test_user_ids)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

item_embeddings = model.tgt_model.iid_embedding.weight.data.cpu().numpy()


target_only_dataset_d2d = TargetDomainDataset1(other_df2,target_domain_user_embeddings, item_embeddings)
target_only_dataloader_d2d = DataLoader(target_only_dataset_d2d, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)

overlap_dataset_d2d = OverlapUsersDataset1(overlap_df2,aux_domain_user_embeddings, target_domain_user_embeddings, item_embeddings)
overlap_dataloader_d2d = DataLoader(overlap_dataset_d2d, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)

test_d2d1 = TargetDomainDataset1(test_df2_new, aux_domain_user_embeddings, item_embeddings)
test_loader_d2d1 = DataLoader(test_d2d1, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)

test_d2d2 = TargetDomainDataset2(test_df2_new, aux_domain_user_embeddings, item_embeddings)
test_loader_d2d2 = DataLoader(test_d2d2, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)

item_popularity = all_train_df2['parent_asin'].value_counts().reset_index()
item_popularity.columns = ['parent_asin', 'popularity']

def train_d2d(UserEmbed, ItemEmbed, diff_model, train_loader1, train_loader2, test_loader, DiffusionData1, DiffusionData2, epochs, lr, weight_decay, non_overlap_weight, lamda):
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

        total_loss = 0
        # predicted_features = {}
        
        for user_id, user_feature, item_feature, rating in train_loader1:
        # for user_id, user_feature, item_feature, rating, is_overlap in train_loader:
            user_id, user_feature, item_feature, rating = user_id.to(device), user_feature.to(device), item_feature.to(device), rating.to(device)
            # user = UserEmbed(user_feature)
            # item = ItemEmbed(item_feature)
            user = user_feature
            item = item_feature
            diffusion_loss = DiffModel.diffusion_loss(diff_model, user, device).mean()
            predicted_feature = DiffModel.p_sample(diff_model, user, device)
            predicted_rating = (predicted_feature * item).sum(1)
            loss = criterion(predicted_rating, rating.float())
            total_loss = (lamda * diffusion_loss + loss) * non_overlap_weight
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        for user_id, user_feature_s, user_feature_t, item_feature, rating in train_loader2:
        # for user_id, user_feature, item_feature, rating, is_overlap in train_loader:
            user_id, user_feature_s, user_feature_t, item_feature, rating = user_id.to(device), user_feature_s.to(device), user_feature_t.to(device), item_feature.to(device), rating.to(device)
            # source = UserEmbed(user_feature_s)
            # target = UserEmbed(user_feature_t)
            # item = ItemEmbed(item_feature)
            source = user_feature_s
            target = user_feature_t
            item = item_feature
            diffusion_loss = DiffModel.diffusion_loss(diff_model, target, device, source).mean()
            predicted_feature = DiffModel.p_sample(diff_model, source, device, source)
            predicted_rating = (predicted_feature * item).sum(1)
            loss = criterion(predicted_rating, rating.float())
            total_loss = lamda * diffusion_loss + loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        UserEmbed.eval()
        ItemEmbed.eval()
        diff_model.eval()

        actuals = []
        predictions = []
        with torch.no_grad():
            
            for user_id,x0, item_feature, rating in test_loader:
                user_id,x0, item_feature, rating = user_id.to(device), x0.to(device), item_feature.to(device), rating.to(device)
                
                # user = UserEmbed(x0)
                # item = ItemEmbed(item_feature)
                user = x0
                item = item_feature
                predicted_feature = DiffModel.p_sample(diff_model, user, device, user)
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
        early_stop_info = early_stopping(rmse)
        

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
    rmse, mae = mlp_model.calculate_rmse_mae(actuals, predictions)
    # print(f"RMSE: {rmse:.5f}, MAE: {mae:.5f}", end='\t')
    return results_df, rmse, mae

epochs = 60
non_overlap_weight=0.8
diff_lr = 0.0001

weight_decay=0.0001
dropout = 0.2
mask_rate = 0.0
lamda = 0.4

all_configurations = []
for mask_rate in [0.0]:
    for t in [0]:
        for T in [1]:
            for hidden_dims in [[embedding_dim*2]]:
                print(f'mask_rate: {mask_rate}, t: {t}, T: {T}, hidden_dims: {hidden_dims}')
                UserEmbed = UserEmbedding(input_size=32, hidden_size=64, output_size=32, dropout_rate=dropout).to(device)
                ItemEmbed = ItemEmbedding(input_size=32, hidden_size=64, output_size=32, dropout_rate=dropout).to(device)
                diff_model = DiffModel.DiffCDR(num_steps=T,t=t,  in_features=embedding_dim, diff_mask_rate=mask_rate,
                                    hidden_dims=hidden_dims, dropout=dropout).to(device)
                best_mae, best_rmse, best_epoch, best_model_state1, best_model_state2, best_model_state3 \
                      = train_d2d(UserEmbed, ItemEmbed, diff_model, target_only_dataloader_d2d, overlap_dataloader_d2d, 
                                test_loader_d2d1, overlap_dataloader, target_only_dataloader, epochs=epochs, 
                                lr=diff_lr, weight_decay=weight_decay, non_overlap_weight = non_overlap_weight, lamda=lamda)
                UserEmbed.load_state_dict(best_model_state1)
                ItemEmbed.load_state_dict(best_model_state2)
                diff_model.load_state_dict(best_model_state3)
                
                # result = test_model(UserEmbed, ItemEmbed, diff_model, test_loader_d2d2)
                # ndcg = mlp_model.ndcg(result)
                # novel = mlp_model.novelty(result,item_popularity)
                # tail_rmse, tail_mae = mlp_model.longtail(result,item_popularity)
                print({
                    "mask_rate": mask_rate,
                    "t": t,
                    "T": T,
                    "hidden_dims": hidden_dims,
                    "best_mae": best_mae,
                    "best_rmse": best_rmse,
                    "best_epoch": best_epoch
                })
                all_configurations.append({
                    "mask_rate": mask_rate,
                    "t": t,
                    "T": T,
                    "hidden_dims": hidden_dims,
                    "best_mae": best_mae,
                    "best_rmse": best_rmse,
                    "best_epoch": best_epoch
                })
for config in all_configurations:
    print(config)




# # Creating datasets and dataloaders
# dataset1 = RatingDataset(all_df1)
# dataloader1 = DataLoader(dataset1, batch_size=4096, shuffle=True, num_workers=4, pin_memory=True)

# dataset2 = RatingDataset(train_df2)
# dataloader2 = DataLoader(dataset2, batch_size=4096, shuffle=True, num_workers=8, pin_memory=True)

# dataset3 = RatingDataset(test_df2_new)
# dataloader3 = DataLoader(dataset3, batch_size=4096, shuffle=True, num_workers=8, pin_memory=True)

# # Model instantiation
# model1 = DNNBase(max_user_id + 1, max_item_id + 1, embedding_dim).to(device)
# # model2 = LookupEmbedding(max_user_id + 1, max_item_id + 1, embedding_dim).to(device)


# class EarlyStopping:
#     def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.delta = delta
#         self.best_loss = float('inf')
    
#     def __call__(self, val_loss):
#         score = -val_loss
#         early_stop_info = ""

#         if self.best_score is None:
#             self.best_score = score
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             if self.verbose:
#                 early_stop_info = f"EarlyStopping counter: {self.counter} out of {self.patience}"
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.counter = 0

#         return early_stop_info
    
# # Training function (assuming both domains use the same function)
# def train(model, dataloader, dataloader3, epochs=60, lr=0.01):
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
#     early_stopping = EarlyStopping(patience=10, verbose=True)

    
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         total_count = 0
#         best_loss = float('inf')
#         best_epoch = 0
#         best_model_state = None
#         for inputs, ratings in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
#             inputs, ratings = inputs.to(device), ratings.to(device)
#             # print(inputs.device, ratings.device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, ratings)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item() * ratings.size(0)
#             total_count += ratings.size(0)
#         average_loss = total_loss / total_count
#         print(f"Epoch {epoch+1}, train loss: {average_loss}", end='\t')

#         model.eval()
#         total_loss = 0
#         total_count = 0

#         for inputs, ratings in dataloader3:
#             inputs, ratings = inputs.to(device), ratings.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, ratings)
#             total_loss += loss.item() * ratings.size(0)
#             total_count += ratings.size(0)
#         average_loss = total_loss / total_count
#         print(f"test loss: {average_loss}", end='\t')
        

#         # 检查是否是最佳模型
#         if average_loss < best_loss:
#             best_loss = average_loss
#             best_epoch = epoch
#             best_model_state = copy.deepcopy(model.state_dict())

#         scheduler.step(average_loss)
#         early_stop_info = early_stopping(average_loss)
#         print(early_stop_info)

#         if early_stopping.early_stop:
#             print("Early stopping")
#             break
#     return best_loss, best_epoch, best_model_state

# print('train')
# # Training models (commented out)
# best_loss, best_epoch, best_model_state1 = train(model1, dataloader1, dataloader3)
# # best_loss, best_epoch, best_model_state2 = train(model2, dataloader2)

# # 保存模型状态字典
# torch.save(best_model_state1, path1 + 'model1_state_dict.pth')
# # torch.save(model2.state_dict(), 'model2_state_dict.pth')
