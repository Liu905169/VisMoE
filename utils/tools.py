import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil

from tqdm import tqdm
from einops import rearrange
import base64
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import os

plt.switch_backend('agg')


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    shutil.rmtree(dir_path)


def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric):
    total_loss = []
    total_mae_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
            # model=model.float() #
            batch_x = batch_x.to(args.data_type).to(accelerator.device)
            batch_y = batch_y.to(args.data_type).to(accelerator.device)

            batch_x_mark = batch_x_mark.to(accelerator.device).to(args.data_type)#.float().to(accelerator.device)
            # print(batch_x_mark.dtype)
            batch_y_mark = batch_y_mark.to(args.data_type).to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).to(args.data_type)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).to(args.data_type).to(
                accelerator.device)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs,t = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)## outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)#
                    # outputs,t = model(batch_x, 10, batch_y, batch_y_mark)

            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))
            # outputs=vali_data.inverse_transform(outputs.cpu())
            # batch_y=vali_data.inverse_transform(batch_y.cpu())

            f_dim = -1 if args.features == 'MS' else 0
            if args.model=='TimeGraph' or args.model=='GraphWavenet':
                outputs = outputs[:, -args.pred_len:,:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:,:, f_dim:].to(accelerator.device)
            else:
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            # p_data=preprocess(batch_x).detach().cpu().numpy()
            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            # plt.imshow(np.array(true[0,:,:,0]))
            # plt.tight_layout()
            # plt.savefig(f'res/figs/pred_{i}.png')
            # if (i + 1) % 1000 == 0:
            #     plot_time_series_2(batch_x[0,:,0].detach().cpu().float(),
            #                     outputs[0,:,0].detach().cpu().float(),
            #                     batch_y[0,:,0].detach().cpu().float(),
            #                     [p[0,:,0].detach().cpu().float() for p in t[0]],
            #                     t[1][0,:],f'f_res/test/{i}',)
                                # t[2][0,:,0].detach().cpu().float())
            # print(batch_x.shape,pred.shape,true.shape)
            #
            # if i%25==0:
            #     data=batch_x.detach().cpu()
            #     plot_time_series(data=data[0,:,0,0],pred=pred[0,:,0,0],true=true[0,:,0,0],p_data=p_data[0,:,0,0],save_name=f'{args.model_comment}/test_{i}')
            #     plot_time_series(data=data[0,:,1,0],pred=pred[0,:,1,0],true=true[0,:,1,0],p_data=p_data[0,:,1,0],save_name=f'{args.model_comment}/test_s_{i}')

            #

            loss = criterion(pred, true)

            mae_loss = mae_metric(pred, true)

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)

    model.train()
    return total_loss, total_mae_loss


def test(args, accelerator, model, train_loader, vali_loader, criterion):
    x, _ = train_loader.dataset.last_insample_window()
    y = vali_loader.dataset.timeseries
    x = torch.tensor(x, dtype=torch.float32).to(accelerator.device)
    x = x.unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        B, _, C = x.shape
        dec_inp = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1)
        outputs = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        id_list = np.arange(0, B, args.eval_batch_size)
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):
            outputs[id_list[i]:id_list[i + 1], :, :] = model(
                x[id_list[i]:id_list[i + 1]],
                None,
                dec_inp[id_list[i]:id_list[i + 1]],
                None
            )
        accelerator.wait_for_everyone()
        outputs = accelerator.gather_for_metrics(outputs)
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        pred = outputs
        true = torch.from_numpy(np.array(y)).to(accelerator.device)
        batch_y_mark = torch.ones(true.shape).to(accelerator.device)
        true = accelerator.gather_for_metrics(true)
        batch_y_mark = accelerator.gather_for_metrics(batch_y_mark)

        loss = criterion(x[:, :, 0], args.frequency_map, pred[:, :, 0], true, batch_y_mark)

    model.train()
    return loss


def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    elif args.data_path=='pmes-bay.h5':
        file='graph_pems.text'
        print(file)
    else:
        file = args.data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content

def plot_time_series(data, pred, true ,p_data ,save_name):
    # 数据检查
    data = np.array(data)
    pred = np.array(pred)
    true = np.array(true)
    # p_data=preprocess(data,dim=0)
    
    P = len(data)
    Q = len(pred)
    
    # 创建时间轴
    time_data = np.arange(P+Q)
    time_pred_true = np.arange(P + Q)

    data=np.append(data,true)
    pred=np.append(p_data,pred)
    # print(p_data)
    
    # 创建图像
    plt.figure(figsize=(10, 6))
    
    # 绘制 data, pred 和 true
    plt.plot(time_data, data, label="Data", color="#1f77b4")
    plt.plot(time_pred_true, pred, label="Prediction", color="#ff7f0e")
    # plt.plot(time_pred_true, true, label="True", color="red", marker="s", linestyle="-")
    
    # 添加标签和图例
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Time Series with Data, Prediction, and True Values")
    # plt.show()
    plt.savefig(f'res/figs/{save_name}.png')
    plt.clf()

def preprocess(data):
        # print('hi')
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


def preprocess_zscore(data):
        # print('hi')
        B,T,N,F=data.shape
        tensor = rearrange(data.clone(),'B T N F->(B N F) T')
        for i in range(B*N*F):
            # t=tensor[i,:]
            # non_zero_values=t[t!=0]
            # if len(non_zero_values) == 0:
            #     continue
            # mean_value = non_zero_values.mean()
            # t[t == 0] = mean_value
            # tensor[i,:]=t
            tensor[i,:]=handle_outliers_zscore(tensor[i,:],1)
        tensor = rearrange(tensor,'(B N F) T->B T N F',B=B,N=N)
        return tensor

def handle_outliers_zscore(tensor, threshold=3.0):
    mean = tensor.mean()
    std = tensor.std()
    z_scores = (tensor - mean) / std

    # 标记离群值
    outliers = z_scores.abs() > threshold
    tensor[outliers] = mean
    return tensor

def masked_mse(preds, labels):
    null_val=0.0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mae(preds, labels):
    null_val=0.0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def data_to_base64(data,x=None):
    if x==None:
        x=range(len(data))
    img=plt.figure()
    plt.plot(x,data)
    figfile = BytesIO()
    img.savefig(figfile, format='png')
    img_base = base64.b64encode(figfile.getvalue()).decode('utf-8')
    plt.clf()
    return img_base

def data_to_gaf(data,x=None):
    # if x==None:
    #     x=range(len(data))
    # img=plt.figure()
    # plt.plot(x,data)
    # figfile = BytesIO()
    # img.savefig(figfile, format='png')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    time_series_normalized = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)
    
    # 计算 Gramian Angular Field (GAF)
    n = len(time_series_normalized)
    gaf = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            gaf[i, j] = np.cos(np.arccos(time_series_normalized[i]) - np.arccos(time_series_normalized[j]))
    gaf=(gaf*127.5+127.5).astype(np.uint8)
    img = Image.fromarray(gaf)
    # img.save("pic2.png")
    figfile = BytesIO()
    img.save(figfile, format='png')
    
    img_base = base64.b64encode(figfile.getvalue()).decode('utf-8')
    return img_base

def plot_time_series_2(data, pred, true ,predicts,weights ,save_name,more=None):#,loss
    # 数据检查
    data = np.array(data)
    pred = np.array(pred)
    true = np.array(true)
    # p_data=preprocess(data,dim=0)
    
    P = len(data)
    Q = len(pred)
    
    # 创建时间轴
    time_data = np.arange(P+Q)
    time_pred_true = np.arange(P,P + Q)

    data=np.append(data,true)
    # pred=np.append(p_data,pred)
    # print(p_data)
    
    # 创建图像
    plt.figure(figsize=(10, 6))
    
    # 绘制 data, pred 和 true
    plt.plot(time_data, data, label="Data")#, color="#1f77b4")
    plt.plot(time_pred_true, pred, label="Prediction")#, color="#ff7f0e")
    for i,predict in enumerate(predicts):
        plt.plot(time_pred_true, predict, label=f"head{i}",linestyle='--')
    
    # plt.plot(time_pred_true, true, label="True", color="red", marker="s", linestyle="-")
    if more is not None:
        plt.plot(time_data,more)
    
    # 添加标签和图例
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.title(f"weights:{weights}")#,loss:{loss}")
    # plt.show()
    path=f'res/figs/{save_name}.png'
    div,_=os.path.split(path)
    if not os.path.exists(div):
        os.makedirs(div)
    plt.savefig(path)
    plt.clf()
    
def disclassify_loss(pred,true,predicts,weight):
    mses = torch.stack([torch.mean((p - true)**2, dim=1) for p in predicts], dim=0)
    
    # 找到每个样本对应的最小MSE值
    min_mse = torch.min(mses, dim=0).values  # 沿预测器维度取最小值
    
    # 生成权重矩阵（支持多个预测器同时最小）
    # if min_mse >torch.mean((p - true)**2, dim=1)
    # print(min_mse.shape)
    w = (mses == min_mse).float()
    # print(weight.shape)
    return torch.mean((w - weight) ** 2)

def pred_similarity(x: torch.Tensor)-> torch.Tensor:
    N=x.shape[1]
    N_t=(N+1)*N/2
    x1 = x.unsqueeze(2)  # 形状变为 (b, n, 1, l)
    x2 = x.unsqueeze(1)  # 形状变为 (b, 1, n, l)
    
    # 计算平方差并沿序列维度取平均
    squared_diff = (x1 - x2) ** 2     # 广播后的形状 (b, n, n, l)
    mse = squared_diff.mean(dim=-1)   # 沿最后一个维度求平均 → (b, n, n)
    return torch.sum(torch.sum(torch.triu(mse), dim=(-1, -2)))/N_t

def choose_favor_loss(x: torch.Tensor)-> torch.Tensor:
    exp=x.shape[1]
    x=torch.sum(x,dim=0)
    x=x/torch.sum(x)
    loss=torch.mean((x - 1/exp) ** 2)
    # print(loss)
    return loss

def pred_norm(true,predicts)-> torch.Tensor:
    exp=predicts.shape[1]
    predicts=rearrange(predicts,'b e p->b p e')
    return torch.mean(torch.norm(predicts-true,dim=-1, p=float('inf'))/exp)

