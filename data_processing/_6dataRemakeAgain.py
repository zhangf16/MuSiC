import json
import pandas as pd

name = 'book'

# 读取itemDescription.jsonl文件
item_file_path = "/home/data/zhangfan/multimodal/" + name +"/itemDescription.jsonl"
error_log_path = "/home/data/zhangfan/multimodal/" + name +"/error_download.txt"
csv_path = "/home/data/zhangfan/multimodal/" + name +"/data.csv"
review_file_path = "/home/data/zhangfan/multimodal/" + name +"/userReview.jsonl"

# 从error_download.txt中读取失败的ASINs
with open(error_log_path, 'r') as file:
    failed_asins = {line.strip() for line in file}

# 删除processed_data.csv中相关记录
df_processed = pd.read_csv(csv_path)
df_processed_filtered = df_processed[~df_processed['parent_asin'].isin(failed_asins)]

# 删除评分少于20的用户
user_counts = df_processed_filtered['user_id'].value_counts()
df_processed_filtered = df_processed_filtered[df_processed_filtered['user_id'].isin(user_counts[user_counts >= 20].index)]

# 用户和物品的总数
num_users = df_processed_filtered['user_id'].nunique()
num_items = df_processed_filtered['parent_asin'].nunique()

print(f"总用户数: {num_users}")
print(f"总物品数: {num_items}")
print(f"总评分数: {len(df_processed_filtered)}")

# 统计每个物品的评分数量
item_counts = df_processed_filtered['parent_asin'].value_counts()
item_distribution = item_counts.value_counts().sort_index()

print("\n物品的评分数量分布:")
print(item_distribution)

# 统计每个用户的评分数量
user_counts = df_processed_filtered['user_id'].value_counts()
user_distribution = user_counts.value_counts().sort_index()

print("\n用户的评分数量分布:")
print(user_distribution)

df_processed_filtered.to_csv(csv_path, index=False)

# 创建用户ID和物品ID的唯一值集合
unique_user_ids = set(df_processed_filtered['user_id'].unique())
unique_parent_asins = set(df_processed_filtered['parent_asin'].unique())

# 删除userReview.jsonl中相关记录
with open(review_file_path, 'r') as file:
    lines = file.readlines()

with open(review_file_path, 'w') as file:
    for line in lines:
        data = json.loads(line.strip())
        if data[0] in unique_user_ids and data[1] in unique_parent_asins:
            file.write(json.dumps(data) + '\n')

# 删除itemDescription.jsonl中相关记录
with open(item_file_path, 'r') as file:
    lines = file.readlines()

with open(item_file_path, 'w') as file:
    for line in lines:
        data = json.loads(line.strip())
        if data[0] in unique_parent_asins:
            file.write(json.dumps(data) + '\n')

