import argparse
from data_provider.data_factory import get_dataset
from tqdm import tqdm
from zhipuai import ZhipuAI,APIStatusError
import base64
from io import BytesIO
from matplotlib import pyplot as plt
import json
import os
import pandas as pd
import sys
# from utils.json_tool import try_parse_json_object
from retrying import retry
import re

from transformers import AutoModelForCausalLM

from utils.janus.models import MultiModalityCausalLM, VLChatProcessor
from utils.janus.utils.io import load_pil_images
import torch

from classify import TimeSeriesClassifier
from einops import rearrange

import numpy as np

classifier=TimeSeriesClassifier(window_size=24,periodic_threshold=0.48,half_periodic_threshold=1.2)


vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained("./janusmodel", local_files_only=True,trust_remote_code=True)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    "./janusmodel", local_files_only=True,trust_remote_code=True
).to(torch.bfloat16).cuda().eval()

def tstoUrl(data):
    x = range(len(data))
    img = plt.figure()
    plt.plot(x, data)
    figfile = BytesIO()
    img.savefig(figfile, format='png')
    img_base = base64.b64encode(figfile.getvalue()).decode('utf-8')
    plt.clf()
    return img_base

def save_ans(to_save:dict,path,index=None):
    # print(to_save)
    for name,value in to_save.items():
        f_name=f'{name}_texts.json' if index is None else f'{name}_{index}_texts.json'
        print(os.path.join(path,f_name))
        if index is not None:value=value[:i]
        with open(os.path.join(path,f_name), 'w+') as file:
            json.dump(value, file, indent=4,ensure_ascii=False)
    print('saved!')
            
def save_outs(to_save:dict,path,index=None):
    for name,value in to_save.items():
        # print(name,value)
        # new_columns=range(args.fix_length)
        # new_columns = [str(x) for x in new_columns]
        f_name=f'{name}.csv' if index is None else f'{name}_{index}.csv'
        print(path)
        if index is not None:value=value[:i]
        data=pd.DataFrame(value)#.reindex(columns=new_columns, fill_value=0).fillna(0).astype("int")
        data.to_csv(os.path.join(path,f_name))

def read_ans(to_read:dict,path,index):
    for name in to_read.keys():
        f_name=f'{name}_{index}_texts.json'
        with open(os.path.join(path,f_name), 'r') as file:
            to_read[name]=json.load(file)
    return to_read

def read_outs(to_read:dict,path,index):
    for name in to_read.keys():
        f_name=f'{name}_{index}.csv'
        to_read[name]=pd.read_csv(os.path.join(path,f_name),index_col=0).values.tolist()
    return to_read

def get_index(path):
    index=0
    for f in os.listdir(path=path):
        match = re.match(r"^.*?(\d+)\.csv$", f)
        if match and index<int(match.group(1)):
            index=int(match.group(1))
    return index

def tryReadPic(message,embbeding=True):
    # print(message)
    pil_images = load_pil_images(message)
    prepare_inputs = vl_chat_processor(
        conversations=message,
        images=pil_images,
        force_batchify=True
    ).to(vl_gpt.device)
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=args.fix_length,
        do_sample=False,
        use_cache=True
    )
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer,outputs[0].cpu().tolist(),get_response(answer)

def get_question(text,image_url=None):
    return {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{text}",
            "images": ['data:image,'+image_url],
        } if image_url is not None else {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{text}",
        }

def get_response(ans):
    return {"role": "<|Assistant|>", "content": f"{ans}"}
    
def save_fig(data,file,title=''):
    x = range(len(data))
    img = plt.figure()
    
    plt.plot(x, data)
    plt.vlines(x=(240,366),ymax=data.max(),ymin=data.min(),colors='k',linestyles='dashed')
    # plt.vlines(x=336,ymax=data.max(),ymin=data.min())
    plt.title(title)
    div,_=os.path.split(file)
    if not os.path.exists(div):
        os.makedirs(div)
    img.savefig(file, format='png')
    plt.clf()
    
def get_embedding(context,ctype,model):
    if ctype=='image':
        bs, n = context.shape[0:2]
        images = rearrange(context, "b n c h w -> (b n) c h w")
        # [b x n, T2, D]
        images_embeds = model.aligner(model.vision_model(images))

        # [b x n, T2, D] -> [b, n x T2, D]
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
        return images_embeds
    else:
        context[context < 0] = 0  # ignore the image embeddings
        inputs_embeds = model.language_model.get_input_embeddings()(context)
        return inputs_embeds

parser = argparse.ArgumentParser(description='Time-LLM')

# data loader
parser.add_argument('--data', type=str, default='ECL', help='dataset type')
parser.add_argument('--root_path', type=str, default='dataset/electricity', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='electricity_0_10.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--savepath', type=str, default='./saves/', help='location of result')

parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')

# forecasting task
parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
parser.add_argument('--label_len', type=int, default=96, help='start token length')
parser.add_argument('--pred_len', type=int, default=720, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

parser.add_argument('--step', type=int, default=1, help='comment') #12*6 12
parser.add_argument('--longterm_length', type=int, default=0, help='comment') #12 7
parser.add_argument('--comment', type=str, default='class_type720_0_10', help='comment')
parser.add_argument('--fix_length', type=int, default=512, help='max token of LLM')

args = parser.parse_args()

train_data = get_dataset(args, 'train')
vali_data = get_dataset(args, 'val')
test_data = get_dataset(args, 'test')

# new_columns=range(args.fix_length)
# new_columns = [str(x) for x in new_columns]

print('train:',len(train_data))
print('val:',len(vali_data))
print('test:',len(test_data))

savepath=args.savepath

types=['This sequence exhibits stable periodic fluctuations with identifiable repeatingpatterns.',
       'This sequence shows partial periodicity, but the cycle length or amplitude varies to some extent.',
       'This sequence displays a clear long-term upward/downward trend.',
       'This sequence lacks clear periodicity or trend, exhibiting random or abrupt variations.']
type_dict={
    "periodic":[1,0,0,0],
    "half_periodic":[0,1,0,0],
    "trend":[0,0,1,0],
    "irregular":[0,0,0,1],
}

def translator(data,trans:dict):
    for k,v in trans.items():
        if k in data:
            return v
    return trans.values()[-1]

# with open('templates/long.json', 'r', encoding='utf-8') as file:
#     long_template = json.load(file)
# with open('templates/short.json', 'r', encoding='utf-8') as file:
#     short_template = json.load(file)

subpath=f'{args.data}_{args.seq_len}_{args.freq}_s{args.step}_l{args.longterm_length}_{args.comment}'
folder_path=os.path.join(savepath,subpath)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    os.makedirs(os.path.join(folder_path,'train'))
    os.makedirs(os.path.join(folder_path,'val'))
    os.makedirs(os.path.join(folder_path,'test'))

# datamap={'train':train_data,'val':vali_data,'test':test_data}
datamap={'val':vali_data,'test':test_data}
# client = ZhipuAI(api_key="f4e2e58558f2a1b7f5a83d86cf6ab0af.sA1sZwLXxdmiiiKq")
# print(len(train_data))
for name,dataset in datamap.items():
    counts=np.array([0,0,0,0])
    # texts={
    #     # 'l_type':[]
    # }
    outs={
        'l_type':[],
    }
    index=get_index(os.path.join(folder_path,name))
    if index!=0:
        texts=read_ans(texts,os.path.join(folder_path,name),index)
        outs=read_outs(outs,os.path.join(folder_path,name),index)
        print(index)
    
    try:
        for i in tqdm(range(index,len(dataset))):
            long=[]
            x,y,_,_=dataset[i]
            
            ps=[]
            for t in types:
                conversation=[get_question(t,tstoUrl(x[-args.label_len:,0]))]
                pil_images = load_pil_images(conversation)
                
                prepare_inputs = vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True
                ).to(vl_gpt.device)

                im=get_embedding(context=prepare_inputs['pixel_values'],ctype='image',model=vl_gpt)
                tx=get_embedding(context=prepare_inputs['input_ids'],ctype='text',model=vl_gpt)
                likely=torch.mean(torch.einsum('btd,bld->btl',im,tx))
                ps.append(likely.item())
                
            ps=np.array(ps)
            output = np.zeros_like(ps, dtype=int)
            output[ps == np.max(ps)] = 1
            
            counts+=np.array(output)
            
            outs['l_type'].append(output)
            
            
            long_base = tstoUrl(x[-args.label_len:])
        print(counts)
    except Exception as e:
        # save_ans(texts,os.path.join(folder_path,name),i)
        save_outs(outs,os.path.join(folder_path,name),i)
        print("出现异常")
        raise e
    except KeyboardInterrupt as e:
        # save_ans(texts,os.path.join(folder_path,name),i)
        save_outs(outs,os.path.join(folder_path,name),i)
        print("程序被中断")
        raise e
        
        
    
    # save_ans(texts,os.path.join(folder_path,name))
    save_outs(outs,os.path.join(folder_path,name))

