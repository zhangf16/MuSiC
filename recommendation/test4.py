import torch
import json
import os
import h5py
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from copy import deepcopy
from typing import List, Optional

os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 加载模型和 tokenizer
model = AutoModel.from_pretrained('/root/autodl-tmp/llm/checkpoint-1000', trust_remote_code=True, output_hidden_states=True, torch_dtype=torch.float16)
model = model.to(device='cuda')
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/llm/checkpoint-1000', trust_remote_code=True)
model.eval()


def chat(
        image=None,
        description=None,
        tokenizer=None,
        vision_hidden_states=None,
        max_new_tokens=1024,
        sampling=True,
        max_inp_length=2048,
        **kwargs
    ):
        images = []
        tgt_sizes = []

        cur_msgs = []

        if image is not None:
            if model.config.slice_mode:
                slice_images, image_placeholder = model.get_slice_image_placeholder(
                    image, tokenizer
                )
                cur_msgs.append(image_placeholder)
                for slice_image in slice_images:
                    slice_image = model.transform(slice_image)
                    H, W = slice_image.shape[1:]
                    images.append(model.reshape_by_patch(slice_image))
                    tgt_sizes.append(torch.Tensor([H // model.config.patch_size, W // model.config.patch_size]).type(torch.int32))
            else:
                images.append(model.transform(image))
                cur_msgs.append(
                    tokenizer.im_start
                    + tokenizer.unk_token * model.config.query_num
                    + tokenizer.im_end
                )
        cur_msgs.append(description)
        if tgt_sizes:
            tgt_sizes = torch.vstack(tgt_sizes)
        
        input_ids = tokenizer.encode('\n'.join(cur_msgs), return_tensors='pt').to(model.device).tolist()[0]

        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }

        generation_config.update(
            (k, kwargs[k]) for k in generation_config.keys() & kwargs.keys()
        )

        with torch.inference_mode():
            res, vision_hidden_states = model.generate(
                input_id_list=[input_ids],
                max_inp_length=max_inp_length,
                img_list=[images],
                tgt_sizes=[tgt_sizes],
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                return_vision_hidden_states=True,
                **generation_config
            )
        answer = res[0]

        return answer 


def estimate_image_tokens(image, model, tokenizer):
    if model.config.slice_mode:
        slice_images, image_placeholder = model.get_slice_image_placeholder(image, tokenizer)
        num_slices = len(slice_images)
        placeholder_length = len(tokenizer.tokenize(image_placeholder))
        
        total_tokens = 0
        for slice_image in slice_images:
            slice_image = model.transform(slice_image).unsqueeze(0).to(model.device)
            reshaped_slice_image = model.reshape_by_patch(slice_image).squeeze(0)
            total_tokens += reshaped_slice_image.numel() // (model.config.patch_size ** 2)
        
        total_tokens += placeholder_length
    else:
        image_tensor = model.transform(image).unsqueeze(0).to(model.device)
        reshaped_image_tensor = model.reshape_by_patch(image_tensor).squeeze(0)
        total_tokens = reshaped_image_tensor.numel() // (model.config.patch_size ** 2)
    
    return total_tokens

name = 'music'
jsonl_file_path = f"/root/autodl-tmp/data/{name}/itemDescription.jsonl"
image_folder_path = f"/root/autodl-tmp/data/{name}/images"
save_path = f"/root/autodl-tmp/data/{name}/itemFeatures.hdf5"

print(f"Processing {name} items...")

with open(jsonl_file_path, 'r') as file:
    i = 0
    for line in file:
        i += 1
        data = json.loads(line.strip())
        parent_asin = data[0]
        description = data[1]
        file_extension = os.path.splitext(data[2])[1]
        image_path = os.path.join(image_folder_path, f"{parent_asin}{file_extension}")

        image = Image.open(image_path).convert('RGB')
        # question = 'Please descript the movie "Ong Bak: The Thai Warrior"?'

        outputs = chat(
            image=image,
            description=description,
            tokenizer=tokenizer
        )
        print(outputs)
        
        print(f"Processed {i} items.")
