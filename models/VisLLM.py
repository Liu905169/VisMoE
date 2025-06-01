from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

from einops import rearrange,repeat



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
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
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


        self.normalize_layers = Normalize(configs.enc_in, affine=False)
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        
        self.decoder=FlattenHead(0,self.patch_nums*configs.d_model,self.pred_len,0.1)
        self.layer_num=2
        
        self.output_linear=nn.Linear(configs.d_model,self.pred_len)
        self.moe_output=moe_predictor(seq_len=self.label_len,pred_len=self.pred_len,
                                      layers=nn.ModuleList([
                                        MLP_layer(self.label_len,configs.d_model,self.layer_num),
                                        ST_layer(self.label_len,configs.d_model,self.layer_num,1,3),
                                        Conv_out(self.label_len,configs.d_model,1,3,self.layer_num),
                                        VisOutputAttentionLayer(self.layer_num,configs.d_model,3,self.label_len,d_keys=configs.d_model)
                                      ]))
        
        self.general_linear=nn.Linear(self.seq_len,self.pred_len)
        self.general_mlp=MLP_layer(self.seq_len,configs.d_model,2)
        self.do_patch=False
        
        

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out,t = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :],t

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        
        x_enc = self.normalize_layers(x_enc, 'norm')
        
        if self.do_patch:
            patch,_=self.patch_embedding(rearrange(x_enc,'b s n->b n s'))
            g=self.decoder(patch)
            g=torch.unsqueeze(g,-1)
        else:
            g=self.general_mlp(x_enc)
        weight=x_mark_enc[:,:,0]
        
        if self.do_patch:
            dec_out,_=self.moe_output(patch,weight)
        else:
            dec_out,_=self.moe_output(x_enc,weight)
        
        dec_out=rearrange(self.output_linear(rearrange(dec_out+g,'b s n->b n s')),'b n s->b s n')
        
        dec_out = self.normalize_layers(dec_out, 'denorm')
        t=([],weight,[])

        return dec_out,t      
    
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
        # print(target_embedding.shape)

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
    def __init__(self,layer_num, d_model, n_heads,var_num, d_keys=None, attention_dropout=0.1):
        super(VisOutputAttentionLayer,self).__init__()
        self.attentions=nn.ModuleList([AttentionLayer(var_num, n_heads, d_keys, var_num, attention_dropout) for _ in range(layer_num)])
        self.layer_num=layer_num
        # self.outLayer=FlattenHead(0,var_num*(layer_num+1)*d_llm,out_len,0.8)
        self.out_linear=nn.Linear(var_num*(layer_num+1), d_model)
        
        
    def forward(self, data):
        data=rearrange(data,'b p f->b f p')
        datas=[data]
        for i in range(self.layer_num):
            data=self.attentions[i](data,data,data)
            datas.append(data)
        # print(data.shape)
        data=rearrange(torch.cat(datas,dim=1),'(b v) f p->b v (p f)',v=1)
        data=self.out_linear(data)
        data=rearrange(data,'b f p->b p f')
        # print(data.shape)
        return data
    
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
            r=l(x)
            # print(r.shape)
            ys.append(r)
        y=torch.stack(ys,dim=-1)
        if patch:
            y=torch.unsqueeze(y,-1)
        out=torch.einsum("bpne,be->bpn",y,weight)
        return out,y
        
class Conv_out(nn.Module):
    def __init__(self,seq_len, d_model,fea_dim,kernel_size,layer_num):
        super(Conv_out,self).__init__()
        self.seq_len=seq_len
        self.d_model=d_model
        self.convs=nn.ModuleList([nn.Conv1d(
                in_channels=fea_dim,
                out_channels=fea_dim,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2
            )for _ in range(layer_num-1)])
        self.linear=nn.Linear(seq_len, d_model)
        
    def forward(self,x):
        x=rearrange(x,'b s n->b n s')
        x=self.linear(x)
        for conv in self.convs:
            x=conv(x)
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
    
class ST_layer(nn.Module):
    def __init__(self,seq_len, d_model,layer_num,fea_dim,kernel_size,decomp_size=25):
        super(ST_layer,self).__init__()
        self.decompsition = series_decomp(decomp_size)
        self.conv=Conv_out(seq_len=seq_len,d_model=d_model,fea_dim=fea_dim,kernel_size=kernel_size,layer_num=layer_num)
        self.mlp=MLP_layer(seq_len=seq_len,d_model=d_model,layer_num=layer_num)
        
    def forward(self,x):
        # print(x.shape)
        t,r=self.decompsition(x)
        t=self.mlp(t)
        r=self.mlp(r)
        x=t+r
        return x
        