import torch
import pickle
from torch.utils.data import Dataset, DataLoader
# import DiffModel
# import DiffModel_new as DiffModel
import DiffModel_short as DiffModel
import torch.nn as nn
import torch.nn.functional as F
import gc
import numpy as np
import copy
import time
import pandas as pd
import mlp_model
import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DNNBase(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim)
        self.iid_embedding = torch.nn.Embedding(iid_all + 1, emb_dim)
        self.linear = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        uid_emb = self.uid_embedding(x[:, 0].unsqueeze(1))
        iid_emb = self.iid_embedding(x[:, 1].unsqueeze(1))
        emb = torch.cat([uid_emb, iid_emb], dim=1)
        x = torch.sum(self.linear(emb[:, 0, :]) * emb[:, 1, :], 1)
        return x

class MetaNet(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim):
        super().__init__()
        self.event_K = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(emb_dim, 1, False))
        self.event_softmax = torch.nn.Softmax(dim=1)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(meta_dim, emb_dim * emb_dim))

    def forward(self, emb_fea, seq_index):
        mask = (seq_index == 0).float()
        event_K = self.event_K(emb_fea)
        t = event_K - torch.unsqueeze(mask, 2) * 1e8
        att = self.event_softmax(t)
        his_fea = torch.sum(att * emb_fea, 1)
        output = self.decoder(his_fea)
        return output.squeeze(1)

class DNNBasedModel(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim, meta_dim_0):
        super().__init__()

        self.emb_dim = emb_dim
        self.src_model = DNNBase(uid_all, iid_all, emb_dim)
        self.tgt_model = DNNBase(uid_all, iid_all, emb_dim)
        self.aug_model = DNNBase(uid_all, iid_all, emb_dim)
        self.meta_net = MetaNet(emb_dim, meta_dim_0)
        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)

    def forward(self, x, stage , device, diff_model=None, ss_model=None,la_model=None,is_task=False):
        if stage == 'train_src':
            return self.src_model.forward(x)
        elif stage in ['train_tgt', 'test_tgt']:
            return self.tgt_model.forward(x)
        
    
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

class TestUserFeatureDataset(Dataset):
    def __init__(self, auxiliary_features, target_features, test_user):
        self.auxiliary_features = auxiliary_features
        self.target_features = target_features
        self.test_user = test_user

    def __len__(self):
        return len(self.target_features)

    def __getitem__(self, idx):
        return torch.tensor(self.auxiliary_features[idx]), torch.tensor(self.target_features[idx]), self.test_user[idx]




def train(diff_model, train_overlap_loader, train_other_loader, test_dataloader, non_overlap_weight, weight_decay,diff_lr, epochs=10):
    optimizer_diff = torch.optim.Adam(params = diff_model.parameters(), lr= diff_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_diff, 'min', patience=3)
    early_stopping = EarlyStopping(patience=10, verbose=True)

    best_loss = float('inf')
    best_epoch = 0
    best_model_state = None

    for epoch in range(epochs):
        start_time = time.time()  # 开始时间
        diff_model.train()
        total_loss = 0
        total_count = 0
        
        for aux_features, target_features in train_overlap_loader:
            aux_features, target_features = aux_features.to(device), target_features.to(device)

            diffusion_loss = DiffModel.diffusion_loss(diff_model, target_features, device, aux_features).mean()
            optimizer_diff.zero_grad()
            diffusion_loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(diff_model.parameters(),1.)
            optimizer_diff.step()
            total_loss += diffusion_loss.item()
            total_count += target_features.size(0)

        for target_features in train_other_loader:
            target_features = target_features.to(device)
            diffusion_loss = DiffModel.diffusion_loss(diff_model, target_features, device).mean()
            diffusion_loss = diffusion_loss * non_overlap_weight
            optimizer_diff.zero_grad()
            diffusion_loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(diff_model.parameters(),1.)
            optimizer_diff.step()
            total_loss += diffusion_loss.item()
            total_count += target_features.size(0)
        
        average_loss = total_loss * 32 / total_count
        print(f"Epoch {epoch} train Loss: {average_loss:.5f}", end='\t')
        
        # test
        total_loss = 0
        total_count = 0
        with torch.no_grad():
            for aux_features, target_features, names in test_dataloader:
                aux_features, target_features = aux_features.to(device), target_features.to(device)
                predicted = DiffModel.p_sample(diff_model, aux_features, device, aux_features)
                batch_loss = F.smooth_l1_loss(predicted, target_features)
                total_loss += batch_loss.item()
                total_count += target_features.size(0)  # 计算总的样本数量
                
        average_loss = total_loss / len(test_dataloader)
        print(f"test loss: {average_loss:.5f}", end='\t')

        # 检查是否是最佳模型
        if average_loss < best_loss:
            best_loss = average_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(diff_model.state_dict())

        scheduler.step(average_loss)

        end_time = time.time()  # 结束时间
        epoch_duration = end_time - start_time
        print(f"{epoch_duration:.2f} seconds", end='\t')

        early_stop_info = early_stopping(average_loss)
        print(early_stop_info)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    return best_loss, best_epoch, best_model_state

def test(diff_model, test_dataloader):
    # test
    total_loss = 0
    total_count = 0
    predicted_features={}
    with torch.no_grad():
        for aux_features, target_features, names in test_dataloader:
            aux_features, target_features = aux_features.to(device), target_features.to(device)
            predicted = DiffModel.p_sample(diff_model, aux_features, device, aux_features)
            # predicted = DiffModel.p_sample(diff_model, aux_features, device, aux_features)
            for name, feature in zip(names, predicted):
                predicted_features[name.item()] = feature.cpu().numpy()
            batch_loss = F.smooth_l1_loss(predicted, target_features)
            total_loss += batch_loss.item()
            total_count += target_features.size(0)  # 计算总的样本数量
            
    average_loss = total_loss / len(test_dataloader)
    print(f"test loss: {average_loss:.5f}")
    # with open(path1 + "diffuse_user_feature.pkl", "wb") as f:
    #     pickle.dump(predicted_features, f)
    return predicted_features

def calculate_rmse_mae(actuals, predictions):
    mse = np.mean([(a - p) ** 2 for a, p in zip(actuals, predictions)])
    mae = np.mean([abs(a - p) for a, p in zip(actuals, predictions)])
    rmse = np.sqrt(mse)
    return rmse, mae

def test_model(model, test_loader1, test_loader2):
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

    actuals = []
    predictions = []
    with torch.no_grad():
        for user_feature, item_feature, rating in test_loader2:
            user_feature, item_feature, rating = user_feature.to(device), item_feature.to(device), rating.to(device)
            predicted_rating = model(user_feature.float(), item_feature.float()).sum(1)
            predictions.extend(predicted_rating.tolist())
            actuals.extend(rating.tolist())
    rmse, mae = calculate_rmse_mae(actuals, predictions)
    print(f"RMSE: {rmse}, MAE: {mae}")

    return rmse, mae

def predict(dataloader3, predicted_features, item_embeddings2):
    # 预测分数
    predicted_scores = []
    actural_scores = []
    criterion = nn.MSELoss()

    # 遍历测试数据
    for inputs, ratings in dataloader3:
        inputs, ratings = inputs.to(device), ratings.to(device)
        user_ids = inputs[:, 0]
        item_ids = inputs[:, 1]

        user_features = [predicted_features.get(user_id.item()) for user_id in user_ids]
        item_features = item_embeddings2[item_ids.cpu().numpy()]

        # 将特征向量转换为张量
        user_features_tensor = torch.tensor(np.array(user_features), dtype=torch.float32, device=device)
        item_features_tensor = torch.tensor(np.array(item_features), dtype=torch.float32, device=device)

        # user_features_tensor = torch.tensor(user_features, dtype=torch.float32, device=device)
        # item_features_tensor = torch.tensor(item_features, dtype=torch.float32, device=device)

        # 计算分数
        scores = (user_features_tensor * item_features_tensor).sum(dim=1)
        
        loss = criterion(scores, ratings)

        predicted_scores.append(scores.cpu().detach().numpy())
        actural_scores.append(ratings.cpu().detach().numpy())


    predicted_scores = np.concatenate(predicted_scores)
    actural_scores = np.concatenate(actural_scores)
    rmse, mae = calculate_rmse_mae(actural_scores, predicted_scores)
    print(f"RMSE: {rmse}, MAE: {mae}")

def train_d2d(diff_model, train_loader1, train_loader2, test_loader, DiffusionData1, DiffusionData2, epochs, lr, weight_decay, non_overlap_weight):
    criterion = nn.MSELoss()
    optimizer_diff = torch.optim.Adam(params = diff_model.parameters(), lr= lr, weight_decay=weight_decay)
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_diff, 'min', patience=3)
    early_stopping = EarlyStopping(patience=10, verbose=True)

    best_rmse = float('inf')
    best_mae = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        start_time = time.time()  # 开始时间
        diff_model.train()

        total_loss = 0
        # predicted_features = {}
        # diffusion_losses = {}


        for aux_features, target_features in DiffusionData1:
            aux_features, target_features = aux_features.to(device), target_features.to(device)
            diffusion_loss = DiffModel.diffusion_loss(diff_model, target_features, device, aux_features).mean()
            optimizer_diff.zero_grad()
            diffusion_loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(diff_model.parameters(),1.)
            optimizer_diff.step()

        for target_features in DiffusionData2:
            target_features = target_features.to(device)
            diffusion_loss = DiffModel.diffusion_loss(diff_model, target_features, device).mean()
            diffusion_loss = diffusion_loss * non_overlap_weight
            optimizer_diff.zero_grad()
            diffusion_loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(diff_model.parameters(),1.)
            optimizer_diff.step()
        
        for user_id, user_feature, item_feature, rating in train_loader1:
        # for user_id, user_feature, item_feature, rating, is_overlap in train_loader:
            user_id, user_feature, item_feature, rating = user_id.to(device), user_feature.to(device), item_feature.to(device), rating.to(device)
            predicted_feature = DiffModel.p_sample(diff_model, user_feature, device)
            predicted_rating = (predicted_feature * item_feature).sum(dim=1)
            loss = criterion(predicted_rating, rating.float()) * non_overlap_weight
            optimizer_diff.zero_grad()
            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(diff_model.parameters(),1.)
            optimizer_diff.step()
        
        for user_id, user_feature_s, user_feature_t, item_feature, rating in train_loader2:
        # for user_id, user_feature, item_feature, rating, is_overlap in train_loader:
            user_id, user_feature_s, user_feature_t, item_feature, rating = user_id.to(device), user_feature_s.to(device), user_feature_t.to(device), item_feature.to(device), rating.to(device)
            predicted_feature = DiffModel.p_sample(diff_model, user_feature_s, device, user_feature_s)
            predicted_rating = (predicted_feature * item_feature).sum(dim=1)
            loss = criterion(predicted_rating, rating.float())
            optimizer_diff.zero_grad()
            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(diff_model.parameters(),1.)
            optimizer_diff.step()
        
        diff_model.eval()

        actuals = []
        predictions = []
        with torch.no_grad():
            
            for user_id,x0, item_feature, rating in test_loader:
                user_id,x0, item_feature, rating = user_id.to(device), x0.to(device), item_feature.to(device), rating.to(device)
                
                predicted_feature = DiffModel.p_sample(diff_model, x0, device, x0)
                predicted_rating = (predicted_feature * item_feature).sum(dim=1)
                predictions.extend(predicted_rating.tolist())
                actuals.extend(rating.tolist())
        rmse, mae = calculate_rmse_mae(actuals, predictions)
        print(f"Epoch {epoch}, RMSE: {rmse:.5f}, MAE: {mae:.5f}", end='\t')

        # 检查是否是最佳模型
        if rmse < best_rmse:
            best_mae = mae
            best_rmse = rmse
            best_epoch = epoch
            best_model_state = copy.deepcopy(diff_model.state_dict())


        scheduler1.step(rmse)
        early_stop_info = early_stopping(rmse)
        

        end_time = time.time()  # 结束时间
        epoch_duration = end_time - start_time
        print(f"{epoch_duration:.2f} seconds", end='\t')
        print(early_stop_info)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    return best_mae, best_rmse, best_epoch, best_model_state

def test_model(diff_model, test_loader):
    diff_model.eval()
    actuals = []
    predictions = []

    results_df = pd.DataFrame(columns=['user_id', 'parent_asin', 'actual_rating', 'predicted_rating'])


    with torch.no_grad():
        for user_id,item_id, x0, item_feature, rating in test_loader:
                user_id,item_id, x0, item_feature, rating = user_id.to(device),item_id.to(device), x0.to(device), item_feature.to(device), rating.to(device)
                
                predicted_feature = DiffModel.p_sample(diff_model, x0, device, x0)
                predicted_rating = (predicted_feature * item_feature).sum(dim=1)

                batch_results = pd.DataFrame({
                    'user_id': user_id.cpu().numpy(),
                    'parent_asin': item_id.cpu().numpy(),
                    'actual_rating': rating.cpu().numpy(),
                    'predicted_rating': predicted_rating.cpu().numpy()
                })
                results_df = pd.concat([results_df, batch_results], ignore_index=True)

                predictions.extend(predicted_rating.tolist())
                actuals.extend(rating.tolist())
    rmse, mae = calculate_rmse_mae(actuals, predictions)
    print(f"RMSE: {rmse:.5f}, MAE: {mae:.5f}", end='\t')
    return results_df