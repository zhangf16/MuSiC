import requests
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

name = 'movie'

# 读取itemDescription.jsonl文件
jsonl_file_path = f"/root/autodl-tmp/data/{name}/itemDescription.jsonl"
output_folder = f"/root/autodl-tmp/data/{name}/images"
error_log_path = f"/root/autodl-tmp/data/{name}/error_download.txt"

def download_image(image_url, save_path):
    try:
        response = requests.get(image_url, timeout=10)  # 添加超时
        print(image_url)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                file.write(response.content)
            return True
    except requests.RequestException as e:
        print(f"Error downloading {image_url}: {e}")
    return False

# 用于下载图片的函数
def download_image_wrapper(data):
    parent_asin, image_url = data[0], data[2]
    file_extension = os.path.splitext(image_url)[1]
    save_path = os.path.join(output_folder, f"{parent_asin}{file_extension}")
    if not download_image(image_url, save_path):
        return (parent_asin, image_url)
    return None

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取所有需要下载的图片链接
images_to_download = []

# with open(jsonl_file_path, 'r') as file:
#     for line in file:
#         data = json.loads(line.strip())
#         images_to_download.append(data)

# 获取已下载的图片列表
downloaded_images = set(os.listdir(output_folder))
with open(jsonl_file_path, 'r') as file:
    for line in file:
        data = json.loads(line.strip())
        parent_asin = data[0]
        image_url = data[2]
        file_extension = os.path.splitext(image_url)[1]
        expected_file_name = f"{parent_asin}{file_extension}"
        
        if expected_file_name not in downloaded_images:
            images_to_download.append(data)


total_images = len(images_to_download)
downloaded_count = 0

# 使用多线程下载图片
error_asins = []
with ThreadPoolExecutor(max_workers=30) as executor:  # 可以调整线程数
    future_to_asin = {executor.submit(download_image_wrapper, data): data[0] for data in images_to_download}
    for future in as_completed(future_to_asin):
        asin = future_to_asin[future]
        result = future.result()
        if result:
            error_asins.append(result)
            print(f"Error downloading image for ASIN: {asin}")
        downloaded_count += 1
        print(f"下载进度: {downloaded_count}/{total_images} 完成", end='\r')

print("\n图片第一次下载完成。")

def download_image2(image_url, save_path):
    try:
        response = requests.get(image_url, timeout=100)  # 添加超时
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                file.write(response.content)
            return True
    except requests.RequestException as e:
        print(f"Error downloading {image_url}: {e}")
    return False

# 重新下载这些有问题的图片
retry_errors = []
count = 0
total_errors = len(error_asins)
with ThreadPoolExecutor(max_workers=30) as executor:
    future_to_asin = {executor.submit(download_image2, asin[1], os.path.join(output_folder, f"{asin[0]}{os.path.splitext(asin[1])[1]}")): asin for asin in error_asins}
    for future in as_completed(future_to_asin):
        asin = future_to_asin[future]
        if not future.result():
            retry_errors.append(asin)
            print(f"再次尝试下载失败: {asin[0]} {asin[1]}")
        count += 1
        print(f"下载进度: {count}/{total_errors} 完成", end='\r')

# 如果有错误，保存错误信息
if retry_errors:
    with open(error_log_path, 'w') as file:
        for asin in retry_errors:
            file.write(f"{asin[0]} {asin[1]}\n")
    print(f"Errors logged to {error_log_path}")
