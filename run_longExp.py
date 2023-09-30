import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--printout', type=int, default=1, help='for final print')
parser.add_argument('--test_set', type=int, default=0, help='for final print')
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
parser.add_argument('--model_id', type=str, required=True, default='SKU_1', help='model id')
parser.add_argument('--model', type=str, required=True, default='DLinear',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='SKU_1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, required=True, default='SKU_1.csv', help='data file')
parser.add_argument('--features', type=str, required=True, default='S',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, required=True, default='adspend', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='d',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--scaler', type=str, default='MinMax', help='scaler method')

# forecasting task
parser.add_argument('--seq_len', type=int, required=True, default=90, help='input sequence length')
parser.add_argument('--label_len', type=int, required=True, default=90, help='start token length')
parser.add_argument('--pred_len', type=int, required=True, default=14, help='prediction sequence length')

# LSTM,GRU,RNN
parser.add_argument('--hidden_dim', type=int, default=128, help='num of hidden dim')
parser.add_argument('--num_layers', type=int, default=4, help='num of layers')

# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
parser.add_argument('--moving_avg', type=int, default=7, help='window size of moving average')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=1, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=1, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=7, help='decomposition-kernel')

# FEDformer
parser.add_argument('--version', type=str, default='Wavelets',
                    help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
parser.add_argument('--mode_select', type=str, default='random',
                    help='for FEDformer, there are two mode selection method, options: [random, low]')
parser.add_argument('--modes', type=int, default=128, help='modes to be selected random 64')
parser.add_argument('--L', type=int, default=8, help='ignore level')
parser.add_argument('--base', type=str, default='legendre', help='mwt base')
parser.add_argument('--cross_activation', type=str, default='tanh',
                    help='mwt cross atention activation function tanh or softmax')

# Formers
parser.add_argument('--embed_type', type=int, default=1, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=4, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=4, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=False)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', type=int, default=1, help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=300, help='train epochs')
parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.005, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--para_mes', type=float, default=1.0, help='Change the weight of mse in loss function')
parser.add_argument('--para_var', type=float, default=1.0, help='Change the weight of variance in loss function')
parser.add_argument('--para_dtw', type=float, default=0.0, help='Change the weight of dtw in loss function')
parser.add_argument('--para_sid', type=float, default=0.0, help='Change the weight of sequenceincrements distance in loss function')


# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

Exp = Exp_Main

if args.is_training:
    if args.printout == 0:
        setting = (f"{args.model_id}_{args.model}_sl{args.seq_len}_ll{args.label_len}_"
                        f"pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_"
                        f"dl{args.d_layers}_df{args.d_ff}_fc{args.factor}_eb{args.embed}_"
                        f"dt{args.distil}_{args.des}_")
        ii = 0
        while os.path.exists(setting + str(ii)):
            ii += 1
        setting = setting + str(ii)
    else:
        setting = (f"{args.model}_{args.target}")
        
        
    exp = Exp(args)
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    if not args.train_only:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)

    torch.cuda.empty_cache()







else:
    if args.printout == 0:
        setting = (f"{args.model_id}_{args.model}_sl{args.seq_len}_ll{args.label_len}_"
                        f"pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_"
                        f"dl{args.d_layers}_df{args.d_ff}_fc{args.factor}_eb{args.embed}_"
                        f"dt{args.distil}_{args.des}_")
        ii = 0
        while os.path.exists(setting + str(ii)):
            ii += 1
        setting = setting + str(ii)
    else:
        setting = (f"{args.model}_{args.target}")

    exp = Exp(args)

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)
    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
    torch.cuda.empty_cache()
