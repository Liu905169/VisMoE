import torch
import torch.nn as nn

class GCon(nn.Module):
    def __init__(self,node_num,gc_rate):
        super(GCon,self).__init__()
        gc_num=int(node_num*gc_rate)
        self.node_condense=nn.Linear(node_num,gc_num)