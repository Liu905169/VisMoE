from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4,Dataset_Graph,Dataset_GraphFlat,Dataset_GraphFlatLS,Dataset_ETT_hourLS,Dateset_preprocess,Dataset_Custom_extra,Dataset_ETT_minuteEX,Dataset_ETT_hourEX
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'm4': Dataset_M4,
    'graph': Dataset_Graph,
    'graph_flat': Dataset_GraphFlat,
    'graph_flatls': Dataset_GraphFlatLS,
    'ETTh1ls': Dataset_ETT_hourLS,
    'preprocess':Dateset_preprocess,
    'ex':Dataset_Custom_extra,
    'm1ex':Dataset_ETT_minuteEX,
    'h1ex':Dataset_ETT_hourEX,
}#datasets end with 'ls' have no provider


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data == 'm4':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
    elif args.data=='preprocess':
        data_set = Data(
            root_path=args.root_path,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            flag=flag,
            embs=args.problems,
        )
    elif 'ex' in args.data:
        # print(args.extra_info_file)
        data_set = Data(
            root_path=args.root_path,
            extra_info_file=args.extra_info_file,
            extra_info_path=args.extra_info_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, 720, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns,
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns
        )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

def get_dataset(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        freq = args.freq
    else:
        freq = args.freq

    if args.data == 'm4':
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
    elif args.data.endswith('ls'):
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns,
            step=args.step,
            longterm_length=args.longterm_length
        )
    elif args.data=='preprocess':
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            embs=args.problems
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns,
        )
    return data_set
