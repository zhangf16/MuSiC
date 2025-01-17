import numpy as np
import h5py
import pandas as pd

def load_features(hdf5_path, name):
    data_in_memory = {}
    with h5py.File(hdf5_path, 'r') as file:
        for key in name:
            data_in_memory[key] = file[key][...]
    return data_in_memory

def recommend(user_features, item_features, top_k=10):
    recommendations = {}
    for user_id, user_vector in user_features.items():
        scores = {item_id: np.dot(user_vector, item_vector)
                  for item_id, item_vector in item_features.items()}
        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        recommendations[user_id] = top_items
    return recommendations

# 加载特征
user_features_path = "/home/data/zhangfan/multimodal/movie/userFeatures.hdf5"
item_features_path = "/home/data/zhangfan/multimodal/music/itemFeatures.hdf5"


test_path = "/home/zhangfan/code/multimodal_diffusion/data/data1/data1/domain2_test_data.csv"
test_data = pd.read_csv(test_path)
test_user = test_data['user_id'].unique()
test_item = test_data['parent_asin'].unique()
actual_interactions = test_data.groupby('user_id')['parent_asin'].apply(list).to_dict()

print(len(test_user))
print(len(test_item))

user_features = load_features(user_features_path, test_user)
item_features = load_features(item_features_path, test_item)

print("loaded")

# 生成推荐
top_k = 20
user_recommendations = recommend(user_features, item_features, top_k=top_k)


print('recommended')

def calculate_precision_at_k(recommendations, actual_interactions, k=10):
    """ 计算 Precision@K """
    precisions = []
    for user_id, recommended_items in recommendations.items():
        actual_items = set(actual_interactions[user_id])
        recommended_top_k = set(item[0] for item in recommended_items[:k])
        hits = len(actual_items.intersection(recommended_top_k))
        precisions.append(hits / len(recommended_top_k))
    return sum(precisions) / len(precisions)

def calculate_ndcg(recommendations, actual_interactions, k=10):
    """ 计算 NDCG@K """
    import math

    def dcg(relevant_elements):
        """ 计算折损累积增益 """
        return sum([int(rel) / math.log2(idx + 2) for idx, rel in enumerate(relevant_elements)])

    ndcgs = []
    for user_id, recommended_items in recommendations.items():
        actual_items = set(actual_interactions[user_id])
        relevant_elements = [1 if item[0] in actual_items else 0 for item in recommended_items[:k]]
        ideal_elements = sorted(relevant_elements, reverse=True)

        denominator = dcg(ideal_elements)
        if denominator == 0:
            user_ndcg = 0
        else:
            user_ndcg = dcg(relevant_elements) / denominator
            
        ndcgs.append(user_ndcg)
    return sum(ndcgs) / len(ndcgs)

# 假设 actual_interactions 是一个字典，其中键为 user_id，值为实际交互的物品列表
# 例如：actual_interactions = {'user1': ['item1', 'item3'], 'user2': ['item2', 'item4']}
precision_at_k = calculate_precision_at_k(user_recommendations, actual_interactions, k=top_k)
ndcg_at_k = calculate_ndcg(user_recommendations, actual_interactions, k=top_k)

print(precision_at_k)
print(ndcg_at_k)