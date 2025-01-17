import json
import os

name = 'movie'

# 定义文件路径
jsonl_file_path = f"/root/autodl-tmp/data/{name}/itemDescription.jsonl"
output_folder = f"/root/autodl-tmp/data/{name}/images"
error_log_path = f"/root/autodl-tmp/data/{name}/error_download.txt"

# 读取所有已下载的图片文件名
downloaded_images = set(os.listdir(output_folder))

# 从 itemDescription.jsonl 读取数据并检查图片是否已下载
errors = []
with open(jsonl_file_path, 'r') as file:
    for line in file:
        data = json.loads(line.strip())
        parent_asin = data[0]
        image_url = data[2]
        file_extension = os.path.splitext(image_url)[1]
        expected_file_name = f"{parent_asin}{file_extension}"
        
        if expected_file_name not in downloaded_images:
            errors.append(parent_asin)

# 将错误记录到 error_download.txt 文件
if errors:
    with open(error_log_path, 'w') as error_file:
        for asin in errors:
            error_file.write(asin + '\n')

    print(f"Errors logged to {error_log_path}")
else:
    print("No missing images found.")
