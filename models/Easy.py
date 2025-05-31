import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.l=nn.Linear(configs.seq_len,configs.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x_enc=self.l(x_enc.permute(0, 2, 1).to(torch.bfloat16)).permute(0, 2, 1).to(torch.float)
        # for name, param in self.l.named_parameters():
        #     print(f"Parameter {name}: \n {param.grad}")

        return x_enc