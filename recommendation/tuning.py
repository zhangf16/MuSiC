import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from typing import List, Optional

model = AutoModel.from_pretrained('/root/autodl-tmp/llm/MiniCPM-V/finetune/output/output_minicpmv2/checkpoint-1000', trust_remote_code=True, output_hidden_states=True, torch_dtype=torch.float16)
model = model.to(device='cuda')

tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/llm/MiniCPM-V/finetune/output/output_minicpmv2/checkpoint-1000', trust_remote_code=True)
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

image = Image.open('./assets/airplane.jpeg').convert('RGB')
msgs = [{"role": "user", "content": "Sometimes a Rogue (The Lost Lords, Book 5) From Publishers Weekly Prolific "
                                    "Regency romance author Putney's fifth Lost Lords installment (after No Longer a "
                                    "Gentleman) shows off her expert storytelling skills. Sarah Clarke-Townsend is "
                                    "with her heavily pregnant twin sister, Mariah, the Duchess of Ashton, "
                                    "when the two are set upon by a band of kidnappers. Sarah pretends to be Mariah "
                                    "in order to protect her, claiming to have given birth and left the baby with a "
                                    "wet nurse. Sarah remains stalwart as the kidnappers whisk her to the coast of "
                                    "Ireland, while Mariah's husband hires Bow Street Runner Rob Carmichael to bring "
                                    "her back safely. When Rob finally catches up with Sarah, he is deeply impressed "
                                    "by her tenacity as they make a harrowing trek home. The two are soon wondering "
                                    "whether mutual admiration and desire are enough to build a relationship upon. "
                                    "Readers will especially love Sarah, a beautiful and resilient damsel in "
                                    "distress. (Sept.) From Booklist *Starred Review* Sarah Clarke-Townsend would do "
                                    "anything for her twin sister, Mariah, the Duchess of Ashton. So when a very "
                                    "pregnant and very cranky Mariah pleads with Sarah to accompany her on an early "
                                    "morning carriage ride, Sarah reluctantly agrees. Everything is going without a "
                                    "hitch until the two women arrive at a long-abandoned church on the Ashton "
                                    "estate, and Mariah realizes that it’s time to give birth. Sending their groom "
                                    "off for help, Sarah tries to make Mariah as comfortable as possible when matters "
                                    "suddenly go from bad to worse. A band of dangerous men suddenly turns up with "
                                    "plans on kidnapping the duchess. After safely hiding Mariah away, Sarah presents "
                                    "herself as the duchess and lets the men take her instead. Fortunately for Sarah, "
                                    "the Duke of Ashton’s good friend Rob Carmichael has just arrived for a visit, "
                                    "and finding people is Rob’s professional specialty. Composed of equal measures "
                                    "of dangerous intrigue and potent passion, Putney’s fifth elegantly written "
                                    "installment in her Lost Lords series delivers captivating characters, "
                                    "an impeccably realized Regency setting, and a thrilling plot rich in action and "
                                    "adventure. --John Charles About the Author Mary Jo Putney is a New York Times "
                                    "and USA Today bestselling author who has written over 60 novels and novellas. A "
                                    "ten-time finalist for the Romance Writers of America RITA, she has won the honor "
                                    "twice and is on the RWA Honor Roll for bestselling authors. In 2013 she was "
                                    "awarded the RWA Nora Roberts Lifetime Achievement Award. Though most of her "
                                    "books have been historical romance, she has also published contemporary "
                                    "romances, historical fantasy, and young adult paranormal historicals. She lives "
                                    "in Maryland with her nearest and dearest, both two and four footed. Visit her at "
                                    "maryjoputney.com Read more Books Literature & Fiction Genre Fiction"
            }]

outputs = get_embedding(
    image=image,
    description=msgs[0]["content"],
    tokenizer=tokenizer
)
hidden_states = outputs.hidden_states

first_hidden_states = hidden_states[0].to(dtype=torch.float32).cpu().numpy()
last_hidden_states = hidden_states[-1].to(dtype=torch.float32).cpu().numpy()
first_last_avg_states = (first_hidden_states + last_hidden_states) / 2
sentence_representation = first_last_avg_states.mean(axis=1).squeeze(0)
print(sentence_representation.shape)

res = model.chat(
    image='./assets/airplane.jpeg',
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True, # if sampling=False, beam_search will be used by default
    temperature=0.7,
    # system_prompt='' # pass system_prompt if needed
)
print(res)