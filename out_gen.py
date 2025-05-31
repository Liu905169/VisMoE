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

import numpy as np

classifier=TimeSeriesClassifier(window_size=24,periodic_threshold=0.48,half_periodic_threshold=1.2)


# vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained("./janusmodel", local_files_only=True,trust_remote_code=True)
# tokenizer = vl_chat_processor.tokenizer
# vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
#     "./janusmodel", local_files_only=True,trust_remote_code=True
# ).to(torch.bfloat16).cuda().eval()

# def get_question(text,image_url=None):
#     q={
#           "content": [
#               {
#                   "text": text,
#                   "type": "text"
#               }
#           ],
#           "role": "user"
#       } if image_url==None else {
#           "content": [
#               {
#                   "image_url": {
#                       "url": image_url
#                   },
#                   "type": "image_url"
#               },
#               {
#                   "text": text,
#                   "type": "text"
#               }
#           ],
#           "role": "user"
#       }
#     return q

# def get_response(ans,role='assistant'):
#     r={
#           "content": [
#               {
#                   "text": ans,
#                   "type": "text"
#               }
#           ],
#           "role": role
#       }
#     return r

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

# @retry(stop_max_attempt_number=3, wait_fixed=1000)#,exceptions=APIStatusError
# def tryReadPic(message,embbeding=True):
#     response = client.chat.completions.create(
#         model="glm-4v-flash",
#         messages=message
#     )
#     text = response.choices[0].message.content
#     reply=get_response(response.choices[0].message.content,response.choices[0].message.role)
        
#     if embbeding:
#         response = client.embeddings.create(
#             model="embedding-3",
#             input=json.dumps(text),
#         )
#         out=response.data[0].embedding
#     else:
#         out=None
#     return text,out,reply

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

dataset_description='ETTm1数据集记录了从2016年7月到2018年7月的变压器数据，每一小时记录一次。'
abnoromal_claim='注意数据集中可能存在异常值，你需要忽略异常值。'
trend_keywords='上升，下降，保持平稳，先上升后下降，先下降后上升，平稳上升，平稳下降,震荡'
types=[
    '周期型：图像中的数据存在明显的周期性波动',
    '弱周期型：图像中的数据周期性波动存在但不明显',
    '趋势型：图像具有明显的趋势，不存在明显的波动',
    '其他：不明显属于以上几种类型的'
]
type_description='你是一个时序图像分析专家，请根据以下标准对输入的时序图像进行分类：\n\
    【分类标准】\n\
    1. 周期型（Periodic）：\n\
    - 存在至少3个重复出现的相似波形/模式\n\
    - 相邻波峰/波谷间隔时间差不超过±15%\n\
    - 振幅波动范围相对稳定（最大波动不超过平均振幅的30%）\n\
    2. 趋势型（Trend）：\n\
    - 整体呈现单调递增/递减走向（允许短期波动但幅度<整体趋势的20%）\n\
    - 线性趋势：相关系数|r| > 0.7\n\
    - 非线性趋势：二阶导数符号保持统一\n\
    3. 其他（Irregular）：\n\
    - 同时包含周期和趋势成分（如趋势性上涨的波动）\n\
    - 无主导模式（随机波动占主导）\n\
    - 存在突变点或断点（单点变化幅度>整体范围的50%）\n\
    【分析维度】\n\
    请按以下顺序进行判断：\n\
    1. 波形重复性检测 → 2. 整体斜率评估 → 3. 波动成分分解\n\
    【输出格式】\n\
    结论：<周期型/趋势型/其他>\n\
    理由：\n\
    - 关键特征1：<特征描述>\n\
    - 关键特征2：<特征描述>\n\
    - 排除条件：<不符合其他类型的理由>'
type_dict={
    "periodic":[1,0,0,0],
    "half_periodic":[0,1,0,0],
    "trend":[0,0,1,0],
    "irregular":[0,0,0,1],
}
type_dict_e={
    "周期型":[1,0,0,0],
    "弱周期型":[0,1,0,0],
    "趋势型":[0,0,1,0],
    "其他":[0,0,0,1],
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
    texts={
        # 'l_type':[]
        # 'l_trend':[],
        # 'l_traits':[],
        # 's_trend':[],
    }
    outs={
        'l_type':[],
        # 'l_trend':[],
        # 'l_traits':[],
        # 'l_periods':[],
        # 's_trend':[],
        # 'x':[],
        # 'y':[],
    }
    index=get_index(os.path.join(folder_path,name))
    if index!=0:
        texts=read_ans(texts,os.path.join(folder_path,name),index)
        outs=read_outs(outs,os.path.join(folder_path,name),index)
        print(index)
    
    try:
        j=0
        for i in tqdm(range(index,len(dataset))):
        # for i in tqdm(range(index,100)):
        # if name=='train' and i<4200: continue
            long=[]#long_template.copy()
            # short=[]#short_template.copy()
            
            # x,y,lx,sx,ex,sy,ey,lsx=dataset[i]
            x,y,_,_=dataset[i]
            # y=y[96:,:]
            # print(x.shape)
            res,score=classifier.classify(x[-96:,0])
            # save_fig(np.concatenate((x,y)),os.path.join(os.path.join(folder_path,name),f'pics/{i}.png'),title=f'{res}_{score:.5f}')
            out=type_dict[res]
            counts+=np.array(out)
            
            # texts['l_type'].append(res)
            outs['l_type'].append(out)
            
            
            long_base = tstoUrl(x[-args.label_len:])
            # # long.append(get_question(image_url=long_base,
            # #                          text=f"{dataset_description}数据的起始时间为{sx}，终止时间为{ex},根据这张图片，请你宏观地描述这张图片的最大值，最小值以及异常点"))
            # # text,out,reply =tryReadPic(long,embbeding=True) 
            # # # texts['l_trend'].append(text)
            # # # outs['l_trend'].append(out)
            # # long.append(reply)
        
            # long.append(get_question(image_url=long_base,
            #                          text=f"{dataset_description},该图片包含了其中一段时间的数据折线图,请你根据图片直接给出图片中的时序数据所属类别，不需要理由，数据的类别包括以下四类：{types[0]};{types[1]};{types[2]};{types[3]}；"))
            # text,_,reply =tryReadPic(long,embbeding=True) 
            # out=translator(text,type_dict_e)
            # print(reply)
            # counts+=np.array(out)
            # texts['l_type'].append(text)
            # outs['l_type'].append(out)
            # long.append(reply)
            
            # long_base = tstoUrl(lx)
            # long.append(get_question(text=f"{dataset_description}数据的起始时间为{sx}，终止时间为{sx}，根据这张图片，将该图片分割为3-4个长度相近的区间，给出区间数以及这些区间的开始和结束节点并陈述为何这样划分。"))
            # text,out,reply =tryReadPic(long,embbeding=False) 
            # long.append(reply)
        
            # long.append(get_question(text=f"{dataset_description}数据的起始时间为{sx}，终止时间为{sx},根据你的划分结果，请整理你的回答，依次详细描述每个区间的趋势特征，你的输出结果应当符合如下格式‘区间1：开始节点，结束节点，趋势特征；区间2：开始节点，结束节点，趋势特征；…区间n：开始节点，结束节点，趋势特征；’"))
            # text,out,reply =tryReadPic(long,embbeding=True) 
            # texts['l_periods'].append(text)
            # outs['l_periods'].append(out)
            # long.append(reply)
        
            # short_base = tstoUrl(x)
            # short.append(get_question(text=f"{dataset_description}这是该序列的最近几段时间的趋势图，根据这张图片，请你尽量详细描述这张图片的整体趋势，你可以选择用以下关键词以及关键词组合来描述：{trend_keywords}",
            #                           image_url=short_base))
            # text,out,reply =tryReadPic(short,embbeding=True) 
            # texts['s_trend'].append(text)
            # outs['s_trend'].append(out)
            # short.append(reply)
            
            #
            
            # outs['x'].append(x[:,0])
            # outs['y'].append(y[:,0])
            j+=1
            if j>=50000:
                j=0
                save_outs(outs,os.path.join(folder_path,name),i)
                outs['l_type'].clear()
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

