import os
import json
import torch
import h5py
from PIL import Image
from lavis.models import load_model_and_preprocess

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)

for name in ['book']:

    # itemDescription.jsonl文件路径和图片文件夹路径
    jsonl_file_path = "/home/data/zhangfan/multimodal/" + name +"/itemDescription.jsonl"
    image_folder_path = "/home/data/zhangfan/multimodal/" + name +"/images"
    save_path = f"/home/data/zhangfan/multimodal/{name}/itemFeatures.hdf5"

    # 创建一个空字典来存储特征向量
    features_dict = {}

    print(f"Processing {name} items...")

    with open(jsonl_file_path, 'r') as file, h5py.File(save_path, 'w') as h5file:
    # with open(jsonl_file_path, 'r') as file:
        i = 0
        for line in file:
            i += 1
            data = json.loads(line.strip())
            parent_asin = data[0]
            description = data[1]
            file_extension = os.path.splitext(data[2])[1]
            image_path = os.path.join(image_folder_path, f"{parent_asin}{file_extension}")
            raw_image = Image.open(image_path).convert("RGB")

            # 处理文字和图像数据
            image_input = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            text_input = txt_processors["eval"](description)

            # 提取特征向量
            sample = {"image": image_input, "text_input": [text_input]}
            features_multimodal = model.extract_features(sample)

            h5file.create_dataset(parent_asin, data=features_multimodal.multimodal_embeds[:, 0, :].squeeze(0).cpu().numpy())


            # features_dict[parent_asin] = features_multimodal.multimodal_embeds.squeeze(0).tolist()

            # if i % 10000 == 0:
            #     with open(f'/home/data/zhangfan/multimodal/{name}/item_features_part_{i//10000}.json', 'w') as f:
            #         json.dump(features_dict, f)
            #     features_dict = {}
            
            # 打印进度
            print(f"Processed {i}items.", end='\r')

    # if features_dict:
    #     # 保存最后一批数据
    #     with open(f'/home/data/zhangfan/multimodal/{name}/item_features_part_{(i//10000) + 1}.json', 'w') as f:
    #         json.dump(features_dict, f)

    print(name, end=' ')
    print("All features saved to itemFeatures")

