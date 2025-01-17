import json
import pandas as pd

def run(processed_df, original_review, new_review, new_item, new_csv):

    # 创建用户ID和物品ID的唯一值集合
    unique_user_ids = set(processed_df['user_id'].unique())
    unique_parent_asins = set(processed_df['parent_asin'].unique())
    
    processed_jsonl_data = []
    seen_combinations = set()  # 用于跟踪已经看到的用户-物品-评论组合

    with open(original_review, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            if data['user_id'] in unique_user_ids and data['parent_asin'] in unique_parent_asins:
                combo = (data['user_id'], data['parent_asin'])  # 包含评论文本的组合
                if len(data['text'])<=10:
                    continue
                if combo not in seen_combinations:  # 检查组合是否已处理
                    seen_combinations.add(combo)
                    # 保存为列表形式，只包含值
                    processed_jsonl_data.append([
                        data['user_id'],
                        data['parent_asin'],
                        data['text'],
                        data['rating']
                    ])

    # 将处理后的数据转换为DataFrame，并指定列名
    columns = ['user_id', 'parent_asin', 'text', 'rating']
    df = pd.DataFrame(processed_jsonl_data, columns=columns)

    # 删除评分少于20的用户
    user_counts = df['user_id'].value_counts()
    df = df[df['user_id'].isin(user_counts[user_counts >= 20].index)]

    # 用户和物品的总数
    num_users = df['user_id'].nunique()
    num_items = df['parent_asin'].nunique()

    print(f"总用户数: {num_users}")
    print(f"总物品数: {num_items}")
    print(f"总评分数: {len(df)}")

    # 统计每个物品的评分数量
    item_counts = df['parent_asin'].value_counts()
    item_distribution = item_counts.value_counts().sort_index()

    print("\n物品的评分数量分布:")
    print(item_distribution)

    # 统计每个用户的评分数量
    user_counts = df['user_id'].value_counts()
    user_distribution = user_counts.value_counts().sort_index()

    print("\n用户的评分数量分布:")
    print(user_distribution)
    
    with open(new_review, 'w') as file:
        for index, row in df.iterrows():
            file.write(json.dumps(row.tolist()) + '\n')
        # for item in processed_jsonl_data:
        #     file.write(json.dumps(item) + '\n')

    print(f"处理后的Review文件已保存到 {new_review}")


    # 移除评论列
    df.drop(columns=['text'], inplace=True)
    df.to_csv(new_csv, index=False)

    # 获取保留的parent_asin集合
    kept_parent_asins = set(df['parent_asin'].unique())
    updated_item_description = []
    with open(new_item, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            if data[0] in kept_parent_asins:
                updated_item_description.append(data)

    # 保存更新后的itemDescription.jsonl文件
    with open(new_item, 'w') as file:
        for item in updated_item_description:
            file.write(json.dumps(item) + '\n')

    print(f"更新后的itemDescription文件已保存到 {new_item}")



if __name__ == "__main__":
    
    # JSONL文件路径
    original_review = "/home/data/zhangfan/multimodal/originalData/Books.jsonl"
    # 保存处理后的数据
    new_review = "/home/data/zhangfan/multimodal/book/userReview.jsonl"

    new_item = "/home/data/zhangfan/multimodal/book/itemDescription.jsonl"


    # 加载处理过的CSV文件
    new_csv = "/home/data/zhangfan/multimodal/book/data.csv"
    df_processed = pd.read_csv(new_csv)
    run(df_processed, original_review, new_review, new_item, new_csv)
    