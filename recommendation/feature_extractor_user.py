import os
import json
import torch
import h5py
from lavis.models import load_model_and_preprocess

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model, _, txt_processors = load_model_and_preprocess(
    name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)

name = "book"
user_review_file = "/home/data/zhangfan/multimodal/" + name +"/userReview.jsonl"
output_hdf5_file = f"/home/data/zhangfan/multimodal/{name}/userFeatures.hdf5"


# 汇总每个用户的评论
user_comments = {}
with open(user_review_file, 'r') as file:
    for line in file:
        data = json.loads(line.strip())
        user_id = data[0]
        comment = data[2]
        if user_id not in user_comments:
            user_comments[user_id] = []
        user_comments[user_id].append(comment)

# 提取特征并保存到HDF5文件
with h5py.File(output_hdf5_file, 'w') as h5file:
    i=0
    for user_id, comments in user_comments.items():
        i += 1
        combined_comment = ' '.join(comments)  # 拼接用户的所有评论
        text_input = txt_processors["eval"](combined_comment)
        
        sample = {"text_input": [text_input]}
        features = model.extract_features(sample, mode="text")

        h5file.create_dataset(user_id, data=features.text_embeds[:, 0, :].squeeze(0).cpu().numpy())

        print(f"Processed {i} users.", end='\r')

print("\nAll user features have been saved.")
