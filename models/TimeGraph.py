from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize
from utils.adj_loader import load_adj
from layers.GraphConv import gcn
from utils.tools import preprocess_zscore
from einops import rearrange,repeat
import torch.nn.functional as F
import time

import matplotlib.pyplot as plt
import os
import numpy as np


transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
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
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.graph_mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = EmbeddingMatchLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)#ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        self.graph_attention_layer = ReprogrammingLayer(configs.d_model*3, configs.n_heads, self.d_ff, self.d_llm) #EmbeddingMatchLayer(configs.d_model*3, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)

        _, _, adj_mx = load_adj(configs.adjdata, configs.adjtype)
        self.supports = [torch.tensor(i) for i in adj_mx]
        self.supports_len = len(self.supports)
        self.node_num=self.supports[0].shape[0]

        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # self.output_projection=OutputLinearLayer(self.d_llm,self.node_num,self.pred_len,self.node_num,1,configs.dropout)
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len * self.node_num,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)


        self.gcov=gcn(configs.d_model,configs.d_model,0,self.supports_len,2)

        # gc_nums=int(self.node_num*configs.gc_rate)

        full_connect=torch.ones(( self.node_num, self.node_num))
        self.global_supports=[torch.triu(full_connect),torch.tril(full_connect)]
        self.globalgcov=gcn(configs.d_model,configs.d_model,0,2,2)

        # self.condense_layer=nn.Linear(self.node_num,gc_nums)
        # self.decondense_layer=nn.Linear(gc_nums,self.node_num)

        # self.info_condenser=nn.Linear(self.node_num*configs.d_model*3,configs.d_model)
        # self.output_attention=OutputAttentionLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        # self.output_attention=OutputAttentionLayer(configs.enc_in, configs.n_heads, self.d_ff, self.d_llm)
        self.output_pool=OutputPoolLayer(self.d_llm,self.node_num,self.pred_len,1)
        # self.depactching=nn.Linear(configs.d_model,self.pred_len*configs.enc_in)

        self.embedingLayer=TSEembbedingLayer(in_channels=1,out_channels=1,kernel_size=3,conv_dep=5)#F.conv1d()

        self.i=0
        self.savepath=f'res/figs/{configs.model_comment}'
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

        self.filler=torch.rand((self.pred_len,self.node_num,1))


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        #x_enc:(B T N F)
        # probe=x_enc.detach().cpu().to(torch.float).numpy()
        # x_enc=self.preprocess(x_enc)
        # res=torch.mean(x_enc, dim=1)
        # res=res.repeat(1,self.pred_len,1,1)
        # probe2=x_enc.detach().cpu().to(torch.float).numpy()
        x_enc = self.normalize_layers(x_enc, 'norm')
        # print(x_enc.size())
        # x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        # time_now=time.time()

        supports=self.supports#[]
        # for sup in self.supports:
        #     sup=sup.to(x_enc.device).to(torch.bfloat16)
        #     sup=self.condense_layer(sup).t()
        #     sup=self.condense_layer(sup).t()
        #     supports.append(sup)
        # graph_data=self.calcute_graph(supports[0])
        node_count=supports[0].shape[0]

        # x_enc=rearrange(self.condense_layer(rearrange(x_enc.to(torch.bfloat16),'b t n f->b t f n')),'b t f n->b t n f').to(torch.float)

        B, T, N, F = x_enc.size()
        
        x_info=repeat(rearrange(torch.mean(x_enc,2),'b t f->(b f) t'),'a b->a b c',c=1)

        min_values = torch.min(x_info, dim=1)[0]
        max_values = torch.max(x_info, dim=1)[0]
        medians = torch.median(x_info, dim=1).values
        lags = self.calcute_lags(x_info)
        trends = x_info.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_info.shape[0]): #节点单独处理B*F*N
            # n_num=b//(B*F)
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            # nodes_values_str=str(graph_data[n_num].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps for every nodes in graph given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}"
                "graph statistics: "
                # f"this node number are : {str(n_num)}"
                # f"top 5 near node are : {nodes_values_str}<|<end_prompt>|>"
                f"there are {node_count} nodes in graph<|<end_prompt>|>"
            )

            prompt.append(prompt_)
        
        # time_now=self.time_info(time_now,'信息提取')

        # x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)
        # print(prompt.shape)
        # time_now=self.time_info(time_now,'信息embedding化')

        # source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        time_enc = rearrange(x_enc,'b t n f->(b n) f t').to(torch.bfloat16)#x_enc.permute(0, 2, 1).contiguous()
        time_enc=self.embedingLayer(time_enc)+time_enc
        enc_out, n_vars = self.patch_embedding(time_enc) #b:B*F*N t:patch_num f:d_model
        graph_info=rearrange(enc_out,'(b n) t f->b f n t',n=N)

        # rearrange(x_enc,'(b n) t f->b t n f') #b:B*N t:patch_num n f:d_model
        # enc_out = self.reprogramming_layer(enc_out, source_embeddings) #, source_embeddings
        # time_now=self.time_info(time_now,'时间对齐')
        # print(torch.cuda.memory_allocated()) 

        graph_info=[self.gcov(graph_info,[i.to(x_enc.device).to(torch.bfloat16) for i in supports])
                    ,self.globalgcov(graph_info,[i.to(x_enc.device).to(torch.bfloat16) for i in self.global_supports])
                    ,graph_info]
        graph_info=torch.cat(graph_info,dim=1)
        # print(graph_info.shape)
        # graph_info=rearrange(graph_info,'b f n t->b t (n f)')
        graph_info=rearrange(graph_info,'b f n t->(b n) t f')
        # graph_info=self.info_condenser(graph_info)
        
        # graph_info=rearrange(graph_info,'b n t f->(b n) t f')
        # print(graph_info.shape)
        graph_embeddings = self.graph_mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        graph_info=self.graph_attention_layer(graph_info,graph_embeddings,graph_embeddings) #
        # print(graph_info.shape)
        graph_info=rearrange(graph_info,'(b n) t f->b (t n) f',n=N)
        # time_now=self.time_info(time_now,'空间对齐')
        # time_now=time.time()
        # print(torch.cuda.memory_allocated())
        # torch.cuda.empty_cache()
        # print(prompt_embeddings.shape,graph_info.shape)

        # graph_info=(graph_info-graph_info.mean())/graph_info.std()*prompt_embeddings.std()+prompt_embeddings.mean()
        llama_enc_out = torch.cat([prompt_embeddings, graph_info], dim=1)
        # graph_info.register_hook(lambda grad: print(grad)) 
        
        # bacth_num=32
        # outs=[]
        # for b in range(0,llama_enc_out.shape[0],bacth_num):
        #     # with torch.no_grad:
        #     end=min(b+bacth_num,llama_enc_out.shape[0])
        #     with torch.inference_mode():
        #         outs.append(self.llm_model(inputs_embeds=llama_enc_out[b:end]).last_hidden_state[:, :, :self.d_ff])
        #     # print(b,torch.cuda.memory_allocated(),outs[b].element_size()*outs[b].numel())
        # dec_out=torch.cat(outs, dim=0)
        # with torch.inference_mode():
        #     dec_out=self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state[:, :, :self.d_ff]
        # print("in:",llama_enc_out.shape)
        dec_out=self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        # print(dec_out)
        # dec_out=dec_out.clone()
        # dec_out=llama_enc_out
        
        # print("out:",dec_out.shape)
        # print(dec_out.shape)
        # time_now=self.time_info(time_now,'大模型推理')

        # enc_out=rearrange(enc_out,'(b n) t f->b (n t) f',n=N)
        # print(x_enc.shape)

        # attention
        # x_dec=torch.cat([x_enc,torch.ones([B,self.pred_len,N,F]).to(x_enc.device)],dim=1).to(torch.bfloat16) #torch.cat([x_enc,self.filler.repeat(B,1,1,1).to(x_enc.device)],dim=1).to(torch.bfloat16)
        # # probe=x_dec.detach().cpu().to(torch.float).numpy()
        # x_dec=rearrange(x_dec,'b t n f->b (n t) f')
        # dec_out=rearrange(dec_out,'(b f) l d->b (l f) d',f=n_vars)
        # # print(enc_out.shape,dec_out.shape)
        # dec_out=self.output_attention(x_dec,dec_out,dec_out)
        # dec_out=rearrange(dec_out,'b (n t) f->b t n f',n=N)

        #pool
        dec_out=self.output_pool(dec_out)

        # linear
        # dec_out=self.output_projection(dec_out[:,-self.node_num:,:])
        # dec_out=rearrange(dec_out,'b (n t f)->b t n f',n=N,t=self.pred_len)



        # print(dec_out.shape)
        # dec_out=rearrange(dec_out,'b (n t) f->b n (t f)',n=N)
        # dec_out=rearrange(self.depactching(dec_out),'b n (t f)->b n t f',t=self.pred_len)
        

        # dec_out=rearrange(dec_out,'(b f) l d->b f d l',f=n_vars)

        # dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        # dec_out=rearrange(dec_out,'b f (t n)->b t n f',n=N)

        # dec_out=rearrange(dec_out,'(b n) f t->b t f n',n=N)
        # dec_out=rearrange(self.decondense_layer(dec_out),'b t f n->b t n f')

        dec_out = self.normalize_layers(dec_out, 'denorm')
        
        # print(dec_out.shape)


        # plt.imshow(probe[0,:,:,0])
        # plt.tight_layout()
        # plt.colorbar()
        # plt.savefig(f'res/figs/iter_ali_sp/TGout.png')
        # plt.clf()
        # with open('res/txts/TGout.txt','a+') as f:
        #     print('---------------------------',file=f)
        #     print(probe,file=f)

        # for name, param in self.named_parameters():
        #     print(f"Parameter {name} grad_fn:", param.grad_fn)

        # print('--------------------')

        # for name, param in self.graph_attention_layer.named_parameters():
        #     print(f"Parameter {name}: {param.shape}")

        # for name, param in self.named_parameters():
        #     print(f"Parameter {name}: \n {param.grad}")

        # if self.i%1000==0:
        #     np.savetxt(os.path.join(self.savepath,f'train_before_{self.i}.csv'),probe[0,:,:,0],delimiter=',')
        #     # print(probe[0,-10:,:])
        #     plt.imshow(probe[0,:,:,0])
        #     plt.tight_layout()
        #     plt.colorbar()
        #     plt.savefig(os.path.join(self.savepath,f'train_before_{self.i}.png'))
        #     plt.clf()
        #     plt.imshow(probe2[0,:,:,0])
        #     plt.tight_layout()
        #     plt.colorbar()
        #     plt.savefig(os.path.join(self.savepath,f'train_after_{self.i}.png'))
        #     plt.clf()

        #     # if()
        #     # plot_module(self.graph_attention_layer,'graph_attention_layer_grad_no_llm',self.i)
        #     with open(f'res/graph_attention_layer_grad_no_llm/res_{self.i}.txt', 'w+') as f:
        #         for name, param in self.graph_attention_layer.named_parameters():
        #             if param.grad is not None:
        #                 print(f"Parameter {name}: \n {param.grad}",file=f)
        #             else:
        #                 print(f"Parameter {name}: \n None",file=f)
        # self.i+=1

        # self.time_info(time_now,'模型结束')

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

    def calcute_graph(self,graph,topk=5):
        graph=graph-torch.diag(graph)
        return torch.topk(graph,k=topk,dim=1)[1]
    
    def time_info(self,time_before,task):
        print(f'{task} time cost:{time.time()-time_before:.5f}')
        return time.time()
    
    def normalize(self,tensor):
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / std
    
    def preprocess(self,data):
        B,T,N,F=data.shape
        tensor = rearrange(data.clone(),'B T N F->(B N F) T')
        for i in range(B*N*F):
            t=tensor[i,:]
            non_zero_values=t[t!=0]
            if len(non_zero_values) == 0:
                continue
            mean_value = non_zero_values.mean()
            t[t == 0] = mean_value
            tensor[i,:]=t
        tensor = rearrange(tensor,'(B N F) T->B T N F',B=B,N=N)
        return tensor


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
        

class EmbeddingMatchLayer(nn.Module):
    def __init__(self, d_model, topk, d_keys=None, d_llm=None, dropout=0.1):
        super(EmbeddingMatchLayer, self).__init__()

        self.target_embedding_projection=nn.Linear(d_model, d_keys)
        self.source_embedding_projection=nn.Linear(d_llm, d_keys)

        self.out_projection = nn.Linear(d_keys*topk, d_llm)
        self.dropout = nn.Dropout(dropout)

        self.topk=topk

    def forward(self, target_embedding, source_embedding):
        target_embedding=self.target_embedding_projection(target_embedding)
        source_embedding=self.source_embedding_projection(source_embedding)

        similarity=torch.einsum("bpf,qf->bpq",F.normalize(target_embedding,dim=-1),F.normalize(source_embedding,dim=-1))
        match_embedding=source_embedding[torch.topk(similarity,k=self.topk,dim=-1)[1]]
        match_embedding=rearrange(match_embedding,"b p k f->b p (k f)")

        return self.out_projection(match_embedding)


class OutputAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(OutputAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_model)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)
        # self.score_norm=nn.LayerNorm([])

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
        # scores=F.normalize(scores,p=2,dim=2)
        # print(scores.shape)

        # probe=rearrange(scores,'b h l s->b l (s h)').detach().cpu().to(torch.float).numpy()
        # # print(probe)
        # np.savetxt('res/figs/TimeGraph-METR-LA-ail_att/scores_l_sh.csv',probe[0,:200,:],delimiter=',')
        

        # plt.imshow(probe[0,:200,:])
        # plt.tight_layout()
        # plt.colorbar()
        # plt.savefig(os.path.join('res/figs/TimeGraph-METR-LA-ail_att',f'scores_l_sh_norm.png'))
        # plt.clf()

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        attention_embedding = torch.einsum("bhls,bshe->blhe", A, value_embedding)

        return attention_embedding

class OutputLinearLayer(nn.Module):
    def __init__(self, in_dim,input_length,out_dim,node_num,f_num,dropout=0.1):
         super(OutputLinearLayer, self).__init__()
         self.inner_dim=256
         self.linear1=nn.Linear(in_dim*input_length,self.inner_dim)
         self.linear2=nn.Linear(self.inner_dim,out_dim*node_num*f_num)
         self.flatten = nn.Flatten(start_dim=-2)
         self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x=self.flatten(x)
        x=self.linear2(self.linear1(x))
        x=self.dropout(x)
        return x


class OutputPoolLayer(nn.Module):
    def __init__(self,input_features,head, output_length, output_features):
        super(OutputPoolLayer, self).__init__()
        self.pool=nn.AdaptiveAvgPool1d(128)
        self.timeLinear=nn.Linear(128,output_length)
        self.linear=nn.Linear(input_features,output_features*head)
        self.head=head

    def forward(self, x):
        x=rearrange(self.timeLinear(self.pool(rearrange(x,'b l f->b f l'))),'b f l->b l f')
        # print(x.shape)
        x=rearrange(self.linear(x),'b p (n f)->b p n f',n=self.head)
        return x



class TSEembbedingLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,conv_dep):
        super(TSEembbedingLayer,self).__init__()
        conv=TSGateLayer(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size)

        self.convs=nn.ModuleList([conv for _ in range(conv_dep)])
        self.outLinear=nn.Linear((conv_dep+1)*out_channels,out_channels)

    def forward(self, data):
        datas=[]
        datas.append(data)
        for layer in self.convs:
            data=layer(data)
            datas.append(data)
        data=torch.cat(datas,dim=1)
        data=self.outLinear(data.permute(0, 2, 1)).permute(0, 2, 1)
        return data

class TSGateLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super(TSGateLayer,self).__init__()
        self.conv1=nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=int((kernel_size-1)/2))
        self.conv2=nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=int((kernel_size-1)/2))
        self.drop=nn.Dropout(0.6)

    def forward(self, data):
        return self.drop(self.conv1(data)*F.sigmoid(self.conv2(data)))


       