
from matplotlib import pyplot as plt
import torch

def plot_heatmaps_of_layer(param, layer_name, fig, axes, row_idx):
    # 获取层的参数
    # weights = layer.weight.detach().to(torch.float).cpu().numpy() if layer.weight is not None else None
    # bias = layer.bias.detach().to(torch.float).cpu().numpy() if layer.bias is not None else None
    # print(weights.shape)
    # print(bias.shape)
    # 绘制权重的热力图
    if(param.ndim==1):
        param=param.unsqueeze(1)
    im = axes[row_idx].imshow(param.detach().to(torch.float).cpu().numpy() , cmap='coolwarm', aspect='auto')
    fig.colorbar(im, ax=axes[row_idx])
    axes[row_idx].set_title(f'{layer_name}')
    # if weights is not None:
        

    # # 绘制偏置的热力图
    # if bias is not None:
    #     im = axes[row_idx, 1].imshow(bias.reshape(1, -1), cmap='coolwarm', aspect='auto')
    #     fig.colorbar(im, ax=axes[row_idx, 1])
    #     axes[row_idx, 1].set_title(f'{layer_name} - Bias')

def plot_module(model,file,index):
    # print(model.modules())
    
    layers = [module for module in model.named_parameters()]
    # print(len(layers))
    # print(layers)
    # 创建一个图形网格，每一层的权重和偏置各占一个子图
    fig, axes = plt.subplots(len(layers), 1, figsize=(10, 5 * len(layers)))
    # 遍历模型的所有层，绘制每层的热力图
    i=0
    for name, param in model.named_parameters():
        # print(param.requires_grad)
        # print(param.shape,param)
        plot_heatmaps_of_layer(param, f'Layer {name}', fig, axes, i)
        i+=1
    # 调整布局
    plt.tight_layout()
    plt.savefig(f'res/{file}/{index}.jpg')