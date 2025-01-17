import json
import os
import requests

name = 'movie'

# 读取itemDescription.jsonl文件
jsonl_file_path = "/root/autodl-tmp/data/" + name +"/itemDescription.jsonl"
output_folder = "/root/autodl-tmp/data/" + name +"/images"
error_log_path = "/root/autodl-tmp/data/" + name +"/error_download.txt"


def download_image(image_url, save_path):
    try:
        response = requests.get(image_url, timeout=100)  # 添加超时
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                file.write(response.content)
            return True
    except requests.RequestException as e:
        print(f"Error downloading {image_url}: {e}")
    return False

# 读取有问题的ASIN
with open(error_log_path, 'r') as file:
    error_asins = [line.strip() for line in file.readlines()]

print(error_asins)

# 重新下载这些有问题的图片
retry_errors = []
with open(jsonl_file_path, 'r') as file:
    for line in file:
        data = json.loads(line.strip())
        parent_asin = data[0]
        image_url = data[2]
        if parent_asin in error_asins:
            file_extension = os.path.splitext(image_url)[1]
            save_path = os.path.join(output_folder, f"{parent_asin}{file_extension}")
            if not download_image(image_url, save_path):
                retry_errors.append(parent_asin)
                print(f"再次尝试下载失败: {parent_asin}{image_url}")

# 更新错误日志
with open(error_log_path, 'w') as file:
    for asin in retry_errors:
        file.write(asin + '\n')

if not retry_errors:
    print("所有图片下载成功。")
else:
    print(f"仍有图片下载失败，已更新错误日志 {error_log_path}")
