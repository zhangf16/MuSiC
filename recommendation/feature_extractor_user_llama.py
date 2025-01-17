from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.tokenize import sent_tokenize
import torch
import json
import h5py
import numpy as np

model_id = "/root/autodl-tmp/llm/Meta-Llama-3-8B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    # torch_dtype=torch.float32,
    device_map="auto",
    output_hidden_states=True
)


name = "book"
user_review_file = "/root/autodl-tmp/data/" + name +"/userReview.jsonl"
output_hdf5_file = f"/root/autodl-tmp/data/{name}/userFeatures.hdf5"

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

        # words = comment.split()
        # if len(words) > 512:
        #     words = words[:512]  # 保留前512个单词

        # adjusted_comment = ' '.join(words)

        # if user_id not in user_comments:
        #     user_comments[user_id] = []
        # user_comments[user_id].append(adjusted_comment)

def split_reviews(reviews, tokenizer, max_tokens=2048):
    all_tokens = tokenizer.tokenize(' '.join(reviews))
    # all_tokens = ' '.join(reviews)
    if len(all_tokens) <= max_tokens or len(reviews)==1:
        return [' '.join(reviews)]
    else:
        # 如果tokens超过了限制，则对评论列表进行二分
        mid_index = len(reviews) // 2
        left_chunk = split_reviews(reviews[:mid_index], tokenizer, max_tokens)
        right_chunk = split_reviews(reviews[mid_index:], tokenizer, max_tokens)
        return left_chunk + right_chunk
    

# 提取特征并保存到HDF5文件
with h5py.File(output_hdf5_file, 'w') as h5file:
    i=0
    for user_id, comments in user_comments.items():
        i += 1

        chunks = split_reviews(comments, tokenizer)

        all_sentence_representations = []
        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            first_hidden_states = hidden_states[0].to(dtype=torch.float32).cpu().numpy()
            last_hidden_states = hidden_states[-1].to(dtype=torch.float32).cpu().numpy()
            first_last_avg_states = (first_hidden_states + last_hidden_states) / 2
            sentence_representation = first_last_avg_states.mean(axis=1).squeeze(0)
            all_sentence_representations.append(sentence_representation)
        overall_representation = np.mean(all_sentence_representations, axis=0)

        h5file.create_dataset(user_id, data=overall_representation)


        print(f"Processed {i} users.", end='\r')

print("\nAll user features have been saved.")


