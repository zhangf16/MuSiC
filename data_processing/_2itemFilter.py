import pandas as pd
import json

def extract_image_url(image_data):
    # 优先检查'large'，如果不存在则回退到'720w'
    preferred_keys = ['large', '720w']
    for key in preferred_keys:
        if key in image_data:
            return image_data[key]
    return ''

def run(df_processed, original_mata, new_item):
    # 获取保留的parent_asin
    kept_parent_asins = set(df_processed['parent_asin'].unique())
    none_description_asins = set()

    # 解析并处理JSONL文件
    processed_meta_data = []

    with open(original_mata, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            if data['parent_asin'] in kept_parent_asins:
                # 提取所需信息
                if data['title'] is None:
                    none_description_asins.add(data['parent_asin'])
                    continue
                title = data.get('title', '')
                if data['description'] is None or len(data['description']) < 1:
                    none_description_asins.add(data['parent_asin'])
                    continue
                description = ' '.join(data.get('description', []))
                categories = ' '.join(data.get('categories', []))
                image_url = extract_image_url(data['images'][0]) if data['images'] else ''
                if len(image_url)<2:
                    none_description_asins.add(data['parent_asin'])
                    continue
                # image_url = data['images'][0]['large'] if data['images'] else ''
                # image_url = data['images'][0].get('720w', 'No 720w image available')

                text = title + ' ' + description + ' ' + categories
                processed_meta_data.append([
                    data['parent_asin'],
                    text,
                    image_url
                ])
    print('因内容丢弃物品数：', end='')
    print(len(none_description_asins))
    
    with open(new_item, 'w') as file:
        for item in processed_meta_data:
            file.write(json.dumps(item) + '\n')

    print(f"处理后的meta数据已保存到 {new_item}")

    # 从CSV文件中删除
    df_csv = df_processed[~df_processed['parent_asin'].isin(none_description_asins)]
    return df_csv
    

if __name__ == "__main__":
    
    # JSONL文件路径
    original_mata = "/root/autodl-tmp/data/originalData/meta_Movies_and_TV.jsonl"
    # 保存处理后的数据
    new_item = "/root/autodl-tmp/data/movie/itemDescription.jsonl"

    # 加载处理过的CSV文件
    new_csv = "/root/autodl-tmp/data/movie/data.csv"
    df_processed = pd.read_csv(new_csv)
    df_csv = run(df_processed, original_mata, new_item)
    df_csv.to_csv(new_csv, index=False)