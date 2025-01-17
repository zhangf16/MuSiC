import pandas as pd


def run(original_csv):
    # 加载数据
    df = pd.read_csv(original_csv)

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

    return df
    

if __name__ == "__main__":
    original_csv = "/home/data/zhangfan/multimodal/originalData/Movies_and_TV.csv"
    new_csv = "/home/data/zhangfan/multimodal/movie/data.csv"
    df = run(original_csv)
    df.to_csv(new_csv, index=False)