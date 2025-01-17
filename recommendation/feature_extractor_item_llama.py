import torch
import json
import os
import h5py
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from copy import deepcopy
from typing import List, Optional

# 加载模型和 tokenizer
model = AutoModel.from_pretrained('/root/autodl-tmp/llm/MiniCPM-Llama3-V-2_5', trust_remote_code=True, output_hidden_states=True, torch_dtype=torch.float16)
model = model.to(device='cuda')
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/llm/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
model.eval()

def generate(
        input_id_list=None,
        img_list=None,
        tgt_sizes=None,
        tokenizer=None,
        max_inp_length: Optional[int] = None,
        vision_hidden_states=None,
        return_vision_hidden_states=False,
        **kwargs
    ):

        assert input_id_list is not None
        bs = len(input_id_list)
        if img_list == None:
            img_list = [[] for i in range(bs)]
        assert bs == len(img_list)

        model_inputs = model._process_list(tokenizer, input_id_list, max_inp_length)

        if vision_hidden_states is None:
            pixel_values = []
            for i in range(bs):
                img_inps = []
                for img in img_list[i]:
                    img_inps.append(img.to(model.device))
                if img_inps:
                    pixel_values.append(img_inps)
                else:
                    pixel_values.append([])
            model_inputs["pixel_values"] = pixel_values
            model_inputs['tgt_sizes'] = tgt_sizes
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        with torch.inference_mode():
            (
                model_inputs["inputs_embeds"],
                vision_hidden_states,
            ) = model.get_vllm_embedding(model_inputs)

            result = model.llm(
                input_ids=None,
                inputs_embeds=model_inputs["inputs_embeds"],
                **kwargs
            )

        if return_vision_hidden_states:
            return result, vision_hidden_states

        return result

def get_embedding(
        image,
        description,
        tokenizer,
        vision_hidden_states=None,
        sampling=False,
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

        # input_ids = tokenizer.apply_chat_template([{'content': '\n'.join(cur_msgs)}], tokenize=True, add_generation_prompt=False)
        # input_ids = tokenizer.encode('\n'.join(cur_msgs), return_tensors='pt').to(model.device)
        # input_ids = tokenizer('\n'.join(cur_msgs), return_tensors='pt').input_ids[0].tolist()
        input_ids = tokenizer.encode('\n'.join(cur_msgs), return_tensors='pt').to(model.device).tolist()[0]

        generation_config = {}

        generation_config.update(
            (k, kwargs[k]) for k in generation_config.keys() & kwargs.keys()
        )

        with torch.inference_mode():
            res, vision_hidden_states = generate(
                input_id_list=[input_ids],
                max_inp_length=max_inp_length,
                img_list=[images],
                tgt_sizes=[tgt_sizes],
                tokenizer=tokenizer,
                vision_hidden_states=vision_hidden_states,
                return_vision_hidden_states=True,
                **generation_config
            )
        answer = res

        return answer 

name = 'movie'
jsonl_file_path = f"/root/autodl-tmp/data/{name}/itemDescription.jsonl"
image_folder_path = f"/root/autodl-tmp/data/{name}/images"
save_path = f"/root/autodl-tmp/data/{name}/itemFeatures.hdf5"

print(f"Processing {name} items...")

with open(jsonl_file_path, 'r') as file, h5py.File(save_path, 'a') as h5file:
    i = 0
    for line in file:
        i += 1
        data = json.loads(line.strip())
        parent_asin = data[0]
        if parent_asin in h5file:
            continue
        description = data[1]
        file_extension = os.path.splitext(data[2])[1]
        image_path = os.path.join(image_folder_path, f"{parent_asin}{file_extension}")

        # image = Image.open(image_path).convert('RGB')
        try:
            image = Image.open(image_path).convert('RGB')
        except OSError as e:
            print(f"Error loading image {image_path}: {e}")
            break

        outputs = get_embedding(
            image=image,
            description=description,
            tokenizer=tokenizer
        )
        hidden_states = outputs.hidden_states

        first_hidden_states = hidden_states[0].to(dtype=torch.float32).cpu().numpy()
        last_hidden_states = hidden_states[-1].to(dtype=torch.float32).cpu().numpy()
        first_last_avg_states = (first_hidden_states + last_hidden_states) / 2
        sentence_representation = first_last_avg_states.mean(axis=1).squeeze(0)
        h5file.create_dataset(parent_asin, data=sentence_representation)
        print(f"Processed {i} items.", end='\r')
print(name, end=' ')
print("All features saved to itemFeatures")
