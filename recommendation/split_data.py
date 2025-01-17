import pandas as pd
import numpy as np
import h5py
import random
import os
import pickle
import sys

ratio = 8

# source_path = "/root/autodl-tmp/data/movie/"
# target_path = "/root/autodl-tmp/data/music/"
# # path1 = "/root/autodl-tmp/multimodal_diffusion/testData/movie2music/" + str(ratio)+ '/'
# path1 = "/root/autodl-tmp/multimodal_diffusion/data/movie2music/" + str(ratio)+ '/'

# source_path = "/root/autodl-tmp/data/book/"
# target_path = "/root/autodl-tmp/data/movie/"
# path1 = "/root/autodl-tmp/multimodal_diffusion/data/book2movie/" + str(ratio)+ '/'

source_path = "/root/autodl-tmp/data/book/"
target_path = "/root/autodl-tmp/data/music/"
path1 = "/root/autodl-tmp/multimodal_diffusion/data/book2music/" + str(ratio)+ '/'


file_path1 = source_path + "data.csv"
file_path2 = target_path + "data.csv"
source_user_features_path = source_path + "userFeatures.hdf5"
target_user_features_path = target_path + "userFeatures.hdf5"
target_item_features_path = target_path + "itemFeatures7000.hdf5"

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
test_users = random.sample(co_users, round(ratio * 0.1 * len(co_users)))

train_df2 = overlap_df2[~overlap_df2['user_id'].isin(test_users)]
all_train_data2 = pd.concat([train_df2, other_df2])
item_counts = all_train_data2['parent_asin'].value_counts()
item_counts.to_pickle(path1 + 'item_counts.pkl')
print(len(set(all_train_data2.user_id)))
sys.exit()

test_df2 = overlap_df2[overlap_df2['user_id'].isin(test_users)]
test_df2_new = test_df2[test_df2['parent_asin'].isin(all_train_data2['parent_asin'].unique())]
test_df2_none = test_df2[~test_df2['parent_asin'].isin(all_train_data2['parent_asin'].unique())]

# 保存新的CSV文件（请替换为实际的保存路径）
overlap_df1.to_csv(path1 + 'domain1_overlap.csv', index=False)
other_df2.to_csv(path1 + 'domain2_other.csv', index=False)
train_df2.to_csv(path1 + 'domain2_train_data.csv', index=False)
test_df2.to_csv(path1 + 'domain2_test_data.csv', index=False)
test_df2_new.to_csv(path1 + 'domain2_test_data_new.csv', index=False)
test_df2_none.to_csv(path1 + 'domain2_test_data_none.csv', index=False)


uid_dict = dict(zip(set(tgt.user_id), range(len(set(tgt.user_id)))))
print(len(set(tgt.user_id)))

iid_dict = dict(zip(set(tgt.parent_asin), range(len(set(tgt.parent_asin)))))


all_train_data2_copy = all_train_data2.copy()
all_train_data2_copy['parent_asin'] = all_train_data2_copy['parent_asin'].map(iid_dict)
all_train_data2_copy.to_csv(path1 + 'domain2_all_train_data_copy.csv', index=False)

def create_id_mapping(ids):
    return {id_: idx for idx, id_ in enumerate(ids)}

def save(test_data, user_features_hdf5_path, item_features_hdf5_path, output_hdf5_path):
    with h5py.File(user_features_hdf5_path, 'r') as user_file, \
         h5py.File(item_features_hdf5_path, 'r') as item_file, \
         h5py.File(output_hdf5_path, 'w') as output_file:
        
        test_data_copy = test_data.copy()
        
        # 获取特征维度
        user_feature_shape = user_file[str(test_data_copy['user_id'].iloc[0])].shape
        item_feature_shape = item_file[str(test_data_copy['parent_asin'].iloc[0])].shape
        print(user_feature_shape)
        
        # 创建 HDF5 文件中的数据集
        
        ratings_ds = output_file.create_dataset('ratings', data=test_data_copy['rating'].values)
        user_features_ds = output_file.create_dataset('user_features', (len(test_data_copy), *user_feature_shape), dtype='float32')
        item_features_ds = output_file.create_dataset('item_features', (len(test_data_copy), *item_feature_shape), dtype='float32')

        # 逐行读取数据并写入新的 HDF5 文件
        for i, (user_id, item_id) in enumerate(zip(test_data_copy['user_id'], test_data_copy['parent_asin'])):
            user_feature = np.array(user_file[str(user_id)][...], dtype='float32')
            item_feature = np.array(item_file[str(item_id)][...], dtype='float32')
            
            user_features_ds[i] = user_feature
            item_features_ds[i] = item_feature
        
        

        # 更新test_df2中的ID
        test_data_copy['user_id'] = test_data_copy['user_id'].map(uid_dict)
        test_data_copy['parent_asin'] = test_data_copy['parent_asin'].map(iid_dict)


        user_ids_ds = output_file.create_dataset('user_ids', data=test_data_copy['user_id'].values)
        item_ids_ds = output_file.create_dataset('item_ids', data=test_data_copy['parent_asin'].values)



save(test_df2, source_user_features_path, target_item_features_path, path1 + 'domain2_test_data_feature.hdf5')
save(test_df2_new, source_user_features_path, target_item_features_path, path1 + 'domain2_test_data_feature_new.hdf5')
save(test_df2_none, target_user_features_path, target_item_features_path, path1 + 'domain2_test_data_feature_none.hdf5')
# save(all_train_data2, target_user_features_path, target_item_features_path, path1 + 'domain2_all_train_data_feature_only_target.hdf5')


def save1(test_data, s_user_features_hdf5_path, t_user_features_hdf5_path, item_features_hdf5_path, output_hdf5_path):
    with h5py.File(s_user_features_hdf5_path, 'r') as s_user_file, \
         h5py.File(t_user_features_hdf5_path, 'r') as t_user_file, \
         h5py.File(item_features_hdf5_path, 'r') as item_file, \
         h5py.File(output_hdf5_path, 'w') as output_file:
        
        test_data_copy = test_data.copy()
        
        # 获取特征维度
        user_feature_shape = t_user_file[str(test_data_copy['user_id'].iloc[0])].shape
        item_feature_shape = item_file[str(test_data_copy['parent_asin'].iloc[0])].shape
        print(user_feature_shape)
        
        # 创建 HDF5 文件中的数据集
        
        ratings_ds = output_file.create_dataset('ratings', data=test_data_copy['rating'].values)
        user_features_s = output_file.create_dataset('user_features_s', (len(test_data_copy), *user_feature_shape), dtype='float32')
        user_features_t = output_file.create_dataset('user_features_t', (len(test_data_copy), *user_feature_shape), dtype='float32')
        item_features = output_file.create_dataset('item_features', (len(test_data_copy), *item_feature_shape), dtype='float32')

        # 逐行读取数据并写入新的 HDF5 文件
        for i, (user_id, item_id) in enumerate(zip(test_data_copy['user_id'], test_data_copy['parent_asin'])):
            user_feature_s = np.array(s_user_file[str(user_id)][...], dtype='float32')
            user_feature_t = np.array(t_user_file[str(user_id)][...], dtype='float32')
            item_feature = np.array(item_file[str(item_id)][...], dtype='float32')
            user_features_s[i] = user_feature_s
            user_features_t[i] = user_feature_t
            item_features[i] = item_feature
        
        

        # 更新test_df2中的ID
        test_data_copy['user_id'] = test_data_copy['user_id'].map(uid_dict)
        test_data_copy['parent_asin'] = test_data_copy['parent_asin'].map(iid_dict)

        user_ids_ds = output_file.create_dataset('user_ids', data=test_data_copy['user_id'].values)
        item_ids_ds = output_file.create_dataset('item_ids', data=test_data_copy['parent_asin'].values)

parent_path = os.path.dirname(path1.rstrip('/'))
other_path = parent_path + '/domain2_train_other_feature.hdf5'
# if not os.path.exists(other_path):
save(other_df2, target_user_features_path, target_item_features_path, other_path)

save1(train_df2, source_user_features_path, target_user_features_path, target_item_features_path, path1 + 'domain2_train_overlapping_feature.hdf5')


test_users = set(test_users)
co_users = set(co_users)
train_target_users = tgt_users - test_users
train_source_users = co_users - test_users

with h5py.File(source_user_features_path, 'r') as source, h5py.File(target_user_features_path, 'r') as target:
    

    train_source_user_feature={}
    for user_id in train_source_users:
        train_source_user_feature[uid_dict[user_id]] = np.array(source[str(user_id)][...], dtype='float32')
    
    train_target_user_feature={}
    for user_id in train_target_users:
        train_target_user_feature[uid_dict[user_id]] = np.array(target[str(user_id)][...], dtype='float32')
    # print(train_target_user_feature.keys())
    
    # test_source_user_feature={}
    # for user_id in test_users:
    #     test_source_user_feature[uid_dict[user_id]] = np.array(source[str(user_id)][...], dtype='float32')

    test_target_user_feature={}
    for user_id in test_users:
        test_target_user_feature[uid_dict[user_id]] = np.array(target[str(user_id)][...], dtype='float32')

    # 保存到pickle文件
    with open(path1 + "train_source_user_feature.pkl", "wb") as f:
        pickle.dump(train_source_user_feature, f)

    with open(path1 + "train_target_user_feature.pkl", "wb") as f:
        pickle.dump(train_target_user_feature, f)

    # with open(path1 + "test_source_user_feature.pkl", "wb") as f:
    #     pickle.dump(test_source_user_feature, f)
    
    with open(path1 + "test_target_user_feature.pkl", "wb") as f:
        pickle.dump(test_target_user_feature, f)
    