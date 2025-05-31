from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoModelForCausalLM
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

from utils.viz import plot_module
from utils.tools import data_to_base64,data_to_gaf
from einops import rearrange,repeat
# from zhipuai import ZhipuAI
import matplotlib.pyplot as plt

from utils.janus.models import MultiModalityCausalLM, VLChatProcessor
from utils.janus.utils.io import load_pil_images


transformers.logging.set_verbosity_error()

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.nf=nf
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # print(self.linear.weight.dtype,x.dtype)
        # plt.plot(range(12),self.linear.bias.detach().cpu().float().numpy())
        # plt.imshow(self.linear.bias[:60].detach().cpu().float().numpy())
        # plt.colorbar()
        # plt.savefig('pic.png')
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        # x/=self.nf
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.problem_len=len(configs.problems)
        
        # self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained("./janusmodel", local_files_only=True,trust_remote_code=True)
        # self.tokenizer = self.vl_chat_processor.tokenizer
        
        # self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        #     "./janusmodel", local_files_only=True,trust_remote_code=True
        # )
        # self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()
        # self.emb=self.vl_gpt.language_model.get_input_embeddings()
        
        # print(len(self.tokenizer))
        # for param in self.tokenizer.parameters():
        #     param.requires_grad = False
        # for param in self.vl_gpt.parameters():
        #     param.requires_grad = False
        
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)


        self.normalize_layers = Normalize(configs.enc_in, affine=False)
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        #--------
        # self.i=0
        # self.embedingLayer=TSEembbedingLayer(in_channels=1,out_channels=1,kernel_size=3,conv_dep=5)
        #--------
        #self.decoder=VisOutputConvLayer(3,2048,3,self.problem_len,self.pred_len)
        # self.decoder=VisOutputAttentionLayer(3,2048,3,self.problem_len,self.pred_len,None,0.9)
        self.num_exp=4
        self.decoder=FlattenHead(0,self.patch_nums*configs.d_model,self.pred_len,0.1)#(0,self.problem_len*2048,self.pred_len,0.8)
        
        
        self.output_linear=nn.Linear(configs.d_model,self.pred_len)#nn.Linear(self.label_len,self.pred_len*self.num_exp)#
        self.moe_output=moe_predictor(seq_len=self.label_len,pred_len=self.pred_len,
                                      layers=nn.ModuleList([
                                        #   Linear_out(self.label_len,self.pred_len)
                                          MLP_layer(self.label_len,configs.d_model,2)
                                          for _ in range(self.num_exp)
                                      ]))
        # self.moe_output=moe_predictor(seq_len=self.label_len,pred_len=self.pred_len,
        #                               layers=nn.ModuleList([
        #                                   Conv_out(self.label_len,self.pred_len,1,3)
        #                                   for _ in range(self.num_exp)
        #                               ]))
        # self.moe_output=moe_predictor(seq_len=int(self.label_len/self.patch_len),pred_len=self.pred_len,
        #                               layers=nn.ModuleList([
        #                                   FlattenHead(0,int(self.label_len/self.patch_len)*configs.d_model,self.pred_len,0.1)
        #                                   for _ in range(self.num_exp)
        #                               ]))
        
        # self.trend_linear=nn.Linear(self.seq_len,self.pred_len)
        # self.weight_linear=nn.Linear(configs.d_model*512*self.problem_len,self.num_exp)
        self.weight_linear=nn.Linear(self.seq_len,self.num_exp)
        self.self_attion=InfoSelfAttentionLayer(4096,configs.d_model)
        self.cross_attion=InfoCrossAttentionLayer(configs.d_model,1,configs.d_model)
        self.avg_size=24
        self.avg_pool=nn.AvgPool1d(self.avg_size,stride=1,padding=int((self.avg_size-1)/2))
        self.output_conv=nn.Conv1d(1,self.num_exp,self.avg_size,stride=8,padding=int(((self.pred_len-1)*8-self.seq_len+self.avg_size)/2))
        self.general_linear=nn.Linear(self.seq_len,self.pred_len)
        self.general_mlp=MLP_layer(self.seq_len,configs.d_model,2)
        self.season_mlp=MLP_layer(self.seq_len,configs.d_model,2)
        self.time_embbing=nn.Embedding(7*24,1)
        self.do_patch=False
        
        

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out,t = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :],t

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_info=x_mark_dec[:,:self.seq_len,2]*7+x_mark_dec[:,:self.seq_len,3]#2 d of w 3 h of d
        y_info=x_mark_dec[:,self.seq_len:self.seq_len+self.pred_len,2]*7+x_mark_dec[:,self.seq_len:self.seq_len+self.pred_len,3]
        x_emb=self.time_embbing(x_info.to(torch.long))
        y_emb=self.time_embbing(y_info.to(torch.long))
        
        # print(y_emb.shape)
        
        
        # print(x_mark_dec[0,:14,3])
        # t=x_enc

        # x=x_enc
        # avg=torch.mean(x_enc)
        # x_enc=x_enc*torch.randn(x_enc.shape).to(x_enc.device).to(x_enc.dtype)
        x_enc = self.normalize_layers(x_enc, 'norm')#-x_emb
        
        # s,x_enc=self.decompsition(x_enc)
        # print(x_dec.shape,s.shape)
        # print(x_enc.shape,x_emb.shape)
        if self.do_patch:
            patch,_=self.patch_embedding(rearrange(x_enc,'b s n->b n s'))
            g=self.decoder(patch)
            g=torch.unsqueeze(g,-1)
        else:
            g=self.general_mlp(x_enc)
        weight=x_mark_enc[:,:,0]
        
        # weight=torch.zeros_like(x_mark_enc[:,:,0]).to(x_enc.device).to(x_enc.dtype)#x_mark_enc[:,:,0]
        # weight[:,0]=1
        # print(x_mark_enc)
        
        # print(x_mark_enc.shape)
        
        # print(weight.shape)
        # trend=rearrange(self.avg_pool(rearrange(x_enc,'b s n->b n s')),'b n p->b p n')
        # x_enc=x_enc-trend
        # print(0,torch.mean(x_enc))
        # info=self.emb(x_mark_enc) #info默认值修改
        # print(info.shape)
        # print(1,torch.mean(info))
        # info=self.self_attion(info)
        # print(info)
        # weight=self.weight_linear(rearrange(info,'b q l e->b (q l e)'))#F.softmax(self.weight_linear(rearrange(info,'b q l e->b (q l e)'))/100,dim=1)#
        # min_vals, _ = torch.min(weight, dim=1, keepdim=True)
        # max_vals, _ = torch.max(weight, dim=1, keepdim=True)
        
        # 最小-最大缩放，将x的范围缩放到[0, 1]
        # weight = F.softmax((weight - min_vals) / (max_vals - min_vals),dim=1)
        # print(weight)
        
        # weight = F.softmax(weight ,dim=1)
        
        # weight=torch.tensor([[0,1,0],[0,1,0]]).to(x_enc.device).to(x_enc.dtype)
        # print(2,torch.mean(info))
        # print(torch.mean(info))
        # dec_out=self.cross_attion(info,x_enc)
        # print(x_enc.shape,dec_out.shape)
        # print(3,torch.mean(dec_out))
        
        
        
        # print()
        # muti_out=rearrange(self.output_conv(rearrange(x_enc,'b s n->b n s')),'b (n e) p->b p n e',e=self.num_exp)[:,:self.pred_len,:,:]
        
        # muti_out=rearrange(self.output_linear(rearrange(x_enc[:,-self.label_len:,:],'b s n->b n s')),'b n (p e)->b p n e',e=self.num_exp)#+\
            
        #     # rearrange(self.trend_linear(rearrange(trend,'b s n->b n s')),'b n p->b p n')
        # # print(x_enc.shape)
        # # print(g.shape)
        # dec_out=torch.einsum('bpne,be->bpn',muti_out,weight)+g#+y_emb#+rearrange(self.trend_linear(rearrange(trend,'b s n->b n s')),'b n p->b p n')
        
        if self.do_patch:
            dec_out,muti_out=self.moe_output(patch,weight)
        else:
            dec_out,muti_out=self.moe_output(x_enc,weight)
        
        # dec_out=g+self.season_mlp(s)#+y_emb#
        # # print(dec_out.shape)
        dec_out=rearrange(self.output_linear(rearrange(dec_out+g,'b s n->b n s')),'b n s->b s n')
        # # print(g.shape,dec_out.shape)
        # dec_out+=g#+y_emb#
        
        dec_out = self.normalize_layers(dec_out, 'denorm')
        t=([self.normalize_layers(muti_out[:,:,:,i]+g, 'denorm') for i in range(self.num_exp)],weight,[])

        return dec_out,t#+x

    def encoder(self,x):
        dataset_description='ETTm1数据集记录了从2016年7月到2018年7月的变压器数据，每一小时记录一次。'
        abnoromal_claim='注意数据集中可能存在异常值，你需要忽略异常值。'
        trend_keywords='上升，下降，保持平稳，先上升后下降，先下降后上升，平稳上升，平稳下降,震荡'
        
        B, T, N = x.size()
        # device=x.device
        # dtype=x.dtype
        # print(dtype)
        # y=np.zeros((B*N,4096*self.d_ff))
        # y=torch.tensor(y).to(device).to(dtype)
        x=x.clone().detach().cpu().to(torch.float).numpy()
        x=rearrange(x,'b t n->(b n) t')
        inputs_embeds=[]
        
        for i in range(B*N):
            img_base=data_to_base64(data=x[i,:])
            question=f'{dataset_description}，这是时间序列某段时间的趋势图，请你根据这张图片，请你宏观地描述这张图片的最大值，最小值，突变值'
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{question}",
                    "images": ['data:image,'+img_base],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            
            pil_images = load_pil_images(conversation)
            prepare_input = self.vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True
            ).to(self.vl_gpt.device)
            inputs_embed = self.vl_gpt.prepare_inputs_embeds(**prepare_input)
            inputs_embeds.append(inputs_embed)
            
        inputs_embeds=torch.cat(inputs_embeds)
        # inputs_embeds=torch.cat([self.soft_prompt.unsqueeze(0).expand(inputs_embeds.shape[0], -1, -1), inputs_embeds], dim=1)
        # print(self.soft_prompt.mean())
        # plt.imshow(self.soft_prompt[:,::20].detach().cpu().float().numpy())
        # plt.colorbar()
        # plt.savefig('pic.png')
        # plt.clf()
        
        
        # prompt_embeds = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        # inputs_embeds=
        # run the model to get the 
        # with torch.no_grad():
        # outputs = self.vl_gpt.language_model.generate(
        #     inputs_embeds=inputs_embeds,
        #     # attention_mask=prepare_inputs.attention_mask,
        #     pad_token_id=self.tokenizer.eos_token_id,
        #     bos_token_id=self.tokenizer.bos_token_id,
        #     eos_token_id=self.tokenizer.eos_token_id,
        #     max_new_tokens=512,
        #     do_sample=False,
        #     use_cache=True
        # )
        # print(outputs.shape)
        outputs = self.vl_gpt.language_model(inputs_embeds=inputs_embeds,output_hidden_states=True).hidden_states[1]
        # outputs=torch.sigmoid(outputs)
        # print(self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True))
        # outputs=outputs[:,-self.d_ff:].to(dtype)
        outputs=rearrange(outputs[:,-self.d_ff:,:],'b p f->b (p f)')
        # y[i,:len(outputs[0])]=outputs[0]
        
        # response = self.client.chat.completions.create(
        #     model="glm-4v-flash",
        #     messages=[
        #      {
        #        "role": "user",
        #        "content": [
        #          {
        #            "type": "image_url",
        #            "image_url": {
        #                "url": img_base
        #            }
        #          },
        #          {
        #            "type": "text",
        #            "text": "请描述这张图片的整体趋势以及是否存在异常值"
        #          }
        #        ]
        #      }
        #    ]
        # )
        # text=response.choices[0].message.content
        # # print(text)
        # response = self.client.embeddings.create(
        #     model="embedding-3",
        #     input=text,
        # )
        # y[i,:]=response.data[0].
            
        # y=torch.tensor(y).to(device).to(dtype)
        return rearrange(outputs,'(b n) t->b t n',b=B)
            
    
class VisOutputLayer(nn.Module):
    def __init__(self,llm_size,pre_len,in_channels,out_channels):
        super(VisOutputLayer,self).__init__()
        self.timeLinear=nn.Linear(llm_size,pre_len)
        self.featureLinear=nn.Linear(in_channels,out_channels)

    def forward(self, data):
        data=self.featureLinear(data)
        data=self.timeLinear(rearrange(data,'b t n->b n t'))
        data=rearrange(data,'b n t->b t n')
        return data
    
class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_model)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        _, S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(B, S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(B, S, H, -1)

        out = self.attention(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def attention(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        attention_embedding = torch.einsum("bhls,bshe->blhe", A, value_embedding)

        return attention_embedding
    
class VisOutputAttentionLayer(nn.Module):
    def __init__(self,layer_num, d_llm, n_heads,var_num,out_len, d_keys=None, attention_dropout=0.1):
        super(VisOutputAttentionLayer,self).__init__()
        self.attentions=nn.ModuleList([AttentionLayer(var_num, n_heads, d_keys, var_num, attention_dropout) for _ in range(layer_num)])
        self.layer_num=layer_num
        self.outLayer=FlattenHead(0,var_num*(layer_num+1)*d_llm,out_len,0.8)
        
    def forward(self, data):
        data=rearrange(data,'b p f->b f p')
        datas=[data]
        for i in range(self.layer_num):
            data=self.attentions[i](data,data,data)
            datas.append(data)
        data=rearrange(torch.cat(datas,dim=1),'(b v) f p->b v f p',v=1)
        return self.outLayer(data)

class VisOutputConvLayer(nn.Module):
    def __init__(self,layer_num, d_llm, kernel_size,var_num,out_len):
        super(VisOutputConvLayer,self).__init__()
        self.attentions=nn.ModuleList([nn.Conv1d(in_channels=var_num,out_channels=var_num,kernel_size=kernel_size,padding=int(kernel_size/2)) for _ in range(layer_num)])
        self.layer_num=layer_num
        self.outLayer=FlattenHead(0,var_num*(layer_num+1)*d_llm,out_len,0.8)
        
    def forward(self, data):
        # data=rearrange(data,'b p f->b f p')
        datas=[data]
        for i in range(self.layer_num):
            data=self.attentions[i](data)
            datas.append(data)
        data=rearrange(torch.cat(datas,dim=1),'(b v) p f->b v f p',v=1)
        return self.outLayer(data)
        
class InfoSelfAttentionLayer(nn.Module):
    def __init__(self, d_llm, d_keys, attention_dropout=0.1):
        super(InfoSelfAttentionLayer, self).__init__()

        # d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_llm, d_keys)
        self.key_projection = nn.Linear(d_llm, d_keys)
        self.value_projection = nn.Linear(d_llm, d_keys)
        # self.out_projection = nn.Linear(d_keys * n_heads, d_model)
        # self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)
        self.norm=nn.InstanceNorm2d(2)

    def forward(self, info):
        #in:info batch_size problem_len token_len d_llm
        #out:info batch_size problem_len token_len d_keys
        target_embedding=source_embedding=value_embedding=info
        # B, L, _ = target_embedding.shape
        # _, S, _ = source_embedding.shape
        # H = self.n_heads

        target_embedding = self.query_projection(target_embedding)#.view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding)#.view(B, S, H, -1)
        value_embedding = self.value_projection(value_embedding)#.view(B, S, H, -1)

        out = self.attention(target_embedding, source_embedding, value_embedding)

        # out = out.reshape(B, L, -1)

        return self.norm(out)#self.out_projection(out)

    def attention(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)
        # print(scale)

        scores = torch.einsum("bqle,bqme->bqlm", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        attention_embedding = torch.einsum("bqlm,bqme->bqle", A, value_embedding)

        return attention_embedding      
    
class InfoCrossAttentionLayer(nn.Module):
    def __init__(self,d_fea, d_keys,d_model, attention_dropout=0.1):
        super(InfoCrossAttentionLayer, self).__init__()

        # d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_fea, d_model)
        self.key_projection = nn.Linear(d_keys, d_model)
        self.value_projection = nn.Linear(d_keys, d_model)
        self.out_projection = nn.Linear(d_model, d_fea)
        # self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)
        self.norm=nn.InstanceNorm2d(2)

    def forward(self, target_embedding, source_embedding):
        value_embedding=source_embedding
        #in:info batch_size problem_len token_len d_llm
        #   data batch_size seq_len feature_size(1)
        #out:data batch_size seq_len feature_size(1)
        # B, L, _ = target_embedding.shape
        # _, S, _ = source_embedding.shape
        # H = self.n_heads

        target_embedding = self.query_projection(target_embedding)#.view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding)#.view(B, S, H, -1)
        value_embedding = self.value_projection(value_embedding)#.view(B, S, H, -1)

        out = self.attention(target_embedding, source_embedding, value_embedding)

        # out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def attention(self, target_embedding, source_embedding, value_embedding):
        # print(target_embedding.shape)
        B, S, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("bse,bqle->bqsl", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        attention_embedding = torch.einsum("bqsl,bqle->bse", A, value_embedding)

        return attention_embedding   
    
class moe_predictor(nn.Module):
    def __init__(self,seq_len, pred_len,layers:nn.ModuleList):
        super(moe_predictor,self).__init__()
        self.seq_len=seq_len
        self.num_exp=len(layers)
        self.layers=layers
        
    def forward(self,x,weight,patch=False):
        x=x[:,-self.seq_len:,:]
        ys=[]
        for l in self.layers:
            ys.append(l(x))
        y=torch.stack(ys,dim=-1)
        if patch:
            y=torch.unsqueeze(y,-1)
        out=torch.einsum("bpne,be->bpn",y,weight)
        return out,y
        
class Conv_out(nn.Module):
    def __init__(self,seq_len, pred_len,fea_dim,kernel_size):
        super(Conv_out,self).__init__()
        self.seq_len=seq_len
        self.pred_len=pred_len
        self.conv=nn.Conv1d(
                in_channels=fea_dim,
                out_channels=fea_dim,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2  # 保持长度不变
            )
        self.linear=nn.Linear(seq_len, pred_len)
        
    def forward(self,x):
        x=rearrange(x,'b s n->b n s')
        x=self.conv(x)
        x=self.linear(x)
        x=rearrange(x,'b n p->b p n')
        return x
    
class Linear_out(nn.Module):
    def __init__(self,seq_len, pred_len):
        super(Linear_out,self).__init__()
        self.seq_len=seq_len
        self.pred_len=pred_len
        self.linear=nn.Linear(seq_len, pred_len)
        
    def forward(self,x):
        x=rearrange(x,'b s n->b n s')
        x=self.linear(x)
        x=rearrange(x,'b n p->b p n')
        return x
    
class MLP_layer(nn.Module):
    def __init__(self,seq_len, d_model,layer_num):
        super(MLP_layer,self).__init__()
        self.seq_len=seq_len
        self.d_model=d_model
        self.first_linear=nn.Linear(seq_len, d_model)
        self.layers=nn.ModuleList([nn.Linear(d_model,d_model)for _ in range(layer_num-1)])
        
    def forward(self,x):
        x=rearrange(x,'b s n->b n s')
        # print(x.shape)
        x=F.sigmoid(self.first_linear(x))
        # print(x.shape)
        for l in self.layers:
            # print(x.shape)
            x=F.sigmoid(l(x))
        x=rearrange(x,'b n p->b p n')
        return x