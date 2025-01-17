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
import mlp_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 定义 MLP
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Swish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, user_feature):
        return self.layers(user_feature)

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Swish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, user_feature):
        return self.layers(user_feature)
    
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

path1 = "/root/autodl-tmp/multimodal_diffusion/data/data1/data1/"
with open(path1 + "train_source_user_feature.pkl", "rb") as f:
    train_source_user_feature = pickle.load(f)

with open(path1 + "train_target_user_feature.pkl", "rb") as f:
    train_target_user_feature = pickle.load(f)

with open(path1 + "test_source_user_feature.pkl", "rb") as f:
    test_source_user_feature = pickle.load(f)

with open(path1 + "test_target_user_feature.pkl", "rb") as f:
    test_target_user_feature = pickle.load(f)

train_source_user = list(train_source_user_feature.keys())
train_target_user = list(train_target_user_feature.keys())
other_user = [uid for uid in train_target_user if uid not in train_source_user]

aux_features = [train_source_user_feature[uid] for uid in train_source_user]
tgt_features = [train_target_user_feature[uid] for uid in train_source_user]

other_features = [train_target_user_feature[uid] for uid in other_user]

# 创建数据集和数据加载器
train_overlap = UserFeatureDataset(train_source_user, aux_features, tgt_features)
train_other = UserFeatureDataset2(other_user, other_features)

train_overlap_loader = DataLoader(train_overlap, batch_size=32, shuffle=True, num_workers=4)
train_other_loader = DataLoader(train_other, batch_size=32, shuffle=True, num_workers=4)

# train_source_user = list(train_source_user_feature.keys())
# train_target_user = list(train_target_user_feature.keys())
test_user = list(test_target_user_feature.keys())

# is_overlap = [uid in train_source_user for uid in train_target_user]

# train_aux_features = [torch.tensor(train_source_user_feature.get(uid, np.zeros_like(next(iter(train_source_user_feature.values()))))).float() for uid in train_target_user]

# train_aux_features = [train_source_user_feature.get(uid, torch.zeros_like(torch.tensor(next(iter(train_source_user_feature.values()))))) for uid in train_target_user]
# train_tgt_features = [train_target_user_feature[uid] for uid in train_target_user]

test_aux_features = [test_source_user_feature[uid] for uid in test_user]
test_tgt_features = [test_target_user_feature[uid] for uid in test_user]

# 创建数据集和数据加载器
# dataset = UserFeatureDataset(train_aux_features, train_tgt_features, is_overlap)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

test_dataset = TestUserFeatureDataset(test_aux_features, test_tgt_features, test_user)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)

test_hdf5_path = path1 + "domain2_test_data_feature_new.hdf5"
test_dataset1 = mlp_model.RatingsDataset(test_hdf5_path)
test_loader1 = DataLoader(test_dataset1, batch_size=256, shuffle=False)

def train(diff_model, train_overlap_loader, train_other_loader, test_dataloader, non_overlap_weight, weight_decay, epochs=10):
    optimizer_diff = torch.optim.Adam(params = diff_model.parameters(), lr= diff_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_diff, 'min', patience=5)
    early_stopping = EarlyStopping(patience=10, verbose=True)

    best_loss = float('inf')
    best_epoch = 0
    best_model_state = None

    for epoch in range(epochs):
        start_time = time.time()  # 开始时间
        diff_model.train()
        total_loss = 0
        total_count = 0
        # for aux_features, target_features, is_overlap in dataloader:
        #     aux_features, target_features, is_overlap = aux_features.to(device), target_features.to(device), is_overlap.to(device)

        #     loss = DiffModel.diffusion_loss(diff_model, target_features, device, aux_features)
        #     weights = torch.where(is_overlap, 1.0, non_overlap_weight).to(device)
        #     weights = weights.unsqueeze(1).expand_as(loss)
        #     weighted_loss = (loss * weights).mean()  # 加权损失
            
        #     optimizer_diff.zero_grad()
        #     weighted_loss.backward()
        #     _ = torch.nn.utils.clip_grad_norm_(diff_model.parameters(),1.)
        #     optimizer_diff.step()
        #     total_loss += weighted_loss.item()
        
        for user_id, aux_features, target_features in train_overlap_loader:
            aux_features, target_features = aux_features.to(device), target_features.to(device)

            diffusion_loss = DiffModel.diffusion_loss(diff_model, target_features, device, aux_features).mean()
            optimizer_diff.zero_grad()
            diffusion_loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(diff_model.parameters(),1.)
            optimizer_diff.step()
            total_loss += diffusion_loss.item()
            total_count += target_features.size(0)

        for user_id, target_features in train_other_loader:
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


diff_lr = 0.000001
epochs = 60
weight_decay = 0.00001

encoder_dim = 768

hidden_dims = [encoder_dim//2]
dropout = 0.2
mask_rate = 0.1
non_overlap_weight = 1.0

all_configurations = []
for t in [0]:
    for T in [5]:
            for hidden_dims in [[encoder_dim//2],[encoder_dim//2, encoder_dim//4], [encoder_dim*2]]:
# for diff_scale in [1]:
#     for mask_rate in [0.1]:#[0.5,0.3,0.1]
#         for non_overlap_weight in [1.0,0.5,0.1,0.0]:
                    print(f't: {t}, T: {T}, hidden_dims: {hidden_dims}')
                    diff_model = DiffModel.DiffCDR(num_steps=1000, T=T,t=t,  in_features=encoder_dim, diff_mask_rate=mask_rate,
                                        hidden_dims=hidden_dims, dropout=dropout).to(device)
                    best_loss, best_epoch, best_model_state = train(diff_model, train_overlap_loader, train_other_loader, test_dataloader, non_overlap_weight, weight_decay,epochs=epochs)
                    
                    diff_model.load_state_dict(best_model_state)
                    predicted_features = test(diff_model, test_dataloader)
                    test_dataset2 = mlp_model.predictDataset(predicted_features, path1 + "domain2_test_data_feature_new.hdf5")
                    test_loader2 = DataLoader(test_dataset2, batch_size=256, shuffle=False)

                    model = mlp_model.MLP(input_dim=768, hidden_dim=128, dropout_rate=0.3).to(device)
                    model_state = torch.load(path1 + 'best_model_state.pth')
                    model.load_state_dict(model_state)
                    
                    rmse, mae = test_model(model, test_loader1, test_loader2)

                    all_configurations.append({
                        "t": t,
                        "T": T,
                        "hidden_dims": hidden_dims,
                        "best_loss": best_loss,
                        "rmse": rmse,
                        "mae": mae,
                        "best_epoch": best_epoch
                    })

                    
# torch.save(diff_model.state_dict(), '/home/zhangfan/code/multimodal_diffusion/save/data1/data1/diffuse_state_dict.pth')

for config in all_configurations:
    print(config)