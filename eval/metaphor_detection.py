import os
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import argparse
import torch
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
import requests
from io import BytesIO
import re
import string
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(0)


model_path = "LLaVA-1.5/checkpoints/memegpt"
model_base = "LLaVA-1.5/llava_1.5"
disable_torch_init()
model_name = get_model_name_from_path("LLaVA-1.5/checkpoints/llava-v1.5-7b-task-lora")
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)



def load_images(image_file):
    out = []
    image = Image.open(image_file).convert("RGB")
    out.append(image)
    return out


image_path_Met = "Meme-DataSet/Met-Meme/Eimages/"
image_path_MemeCap = "Meme-DataSet/MemeCap/memes/"
f_Met_path = "Meme-DataSet/Met-Meme/test_label_E.xlsx"
f_MemeCap_path = "Meme-DataSet/MemeCap/memes-test.json"



file = pd.read_excel(f_Met_path)
co_path = "Met-Meme-Metaphor.xlsx"
co_file = pd.read_excel(co_path)


before_data = []
before_output = pd.read_excel(co_path)
for i in range(len(before_output)):
    image_name = before_output.loc[i, 'image_name']
    before_data.append(image_name)

image_data = []

Meta_question = "Please determine whether this meme contains metaphorical information. Please answer only 'Yes' or 'No'."

for i in range(len(file)):
    image_name = file.loc[i, 'image_name']
    Meta_gt = file.loc[i, 'metaphor']

    if image_name in before_data:
        print(f"当前数据: {image_name} 已经被评测过, 跳过")
        continue

    prompts = f"{Meta_question}"
    qs = prompts
    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates['llava_v1'].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    qs_prompt = conv.get_prompt()

    image_file = os.path.join(image_path_Met, image_name)
    images = load_images(image_file)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
    input_ids = (tokenizer_image_token(qs_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0.1,
            top_p=None,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True,
        )

    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    image_data.append({"image_name": image_name,
                       "Meta_gt": Meta_gt,
                       "prompt": Meta_question,
                       'Meta_test': answer,
                       'comment': 'XXX'})

    if i != 0 and i % 50 == 0:
        output = pd.read_excel(co_path)
        df = pd.DataFrame(image_data)
        output = output._append(df)
        output.to_excel(co_path, index=False)
        print(f"当前处理到第{i}张图片, 保存一下之前的数据")
        image_data = []


output = pd.read_excel(co_path)
df = pd.DataFrame(image_data)
output = output._append(df)
output.to_excel(co_path, index=False)
print(f"评估完成, 数据已保存至: {co_path}")






























