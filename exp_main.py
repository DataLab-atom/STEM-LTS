from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, vali, test
from utils.metrics import metric
from STEM_TEMPO import STEM_TEMPO

import sys
from torch.utils.data import Subset
from numpy.random import choice

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from Clip_loss import ClipLoss

import os
import time
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

class Exp_Main:
    def __init__(self, args,config):
        self.args = args
        self.config = config

        self.device = torch.device('cuda:0')
        self.model = self._get_model()
        params = self.model.parameters()
        self.model_optim = torch.optim.Adam(params, lr=args.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.model_optim, T_max=args.tmax, eta_min=1e-8)
        self.early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        
        self.criterion = self._select_criterion()

        self.train_data, self.train_loader = self._get_data(flag='train')
        self.vali_data, self.vali_loader = self._get_data(flag='val')
        self.test_data, self.test_loader = self._get_data(flag='test')
    


    def _get_model(self):
        model = STEM_TEMPO(self.args, self.device)
        dataset_saved_name_map =  {'ETTm1':'Ettm1','traffic':'traffic','weather':'weather','electricity':'elc'}
        dataset_save_name = dataset_saved_name_map[self.args.datasets]
        state_dict = torch.load(f'saved/{dataset_save_name}/{self.args.model}_6_{self.args.pred_len}_{dataset_save_name}.pth')
        model.load_state_dict(state_dict)
        model.logit_scale = nn.Parameter(torch.tensor(0.001).log())
        model.clip_criterion = ClipLoss()
        device_ids = list(range(torch.cuda.device_count()))  
        model = model.cuda()
        if len(device_ids) > 1:
            model = DataParallel(model, device_ids=device_ids).cuda()
        return model
    
    def _get_data(self, flag):
        config,args = self.config,self.args
        args.data = config['datasets'][args.datasets].data
        args.root_path = config['datasets'][args.datasets].root_path
        args.data_path = config['datasets'][args.datasets].data_path
        args.data_name = config['datasets'][args.datasets].data_name
        args.features = config['datasets'][args.datasets].features
        args.freq = config['datasets'][args.datasets].freq
        args.target = config['datasets'][args.datasets].target
        args.embed = config['datasets'][args.datasets].embed
        args.percent = config['datasets'][args.datasets].percent
        args.lradj = config['datasets'][args.datasets].lradj
        if args.freq == 0:
            args.freq = 'h'

        min_sample_num = sys.maxsize
        data, loader = data_provider(args, flag)
        if not (flag == 'test'):
            data_len = len(data)
            min_sample_num = min(min_sample_num, data_len)
            if args.datasets not in ['ETTh1', 'ETTh2', 'ILI', 'exchange'] and args.equal == 1: 
                data = Subset(data, choice(data_len, min_sample_num))
                loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        return data, loader

    def _select_criterion(self):
        if self.args.loss_func == 'mse':
            criterion = nn.MSELoss()
        elif self.args.loss_func == 'smape':
            class SMAPE(nn.Module):
                def __init__(self):
                    super(SMAPE, self).__init__()
                def forward(self, pred, true):
                    return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
            criterion = SMAPE()
        return criterion

    def train_test(self,iter = 1):
        args,train_steps = self.args,len(self.train_loader)   
        path = os.path.join(args.checkpoints, '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, 336, args.label_len, args.pred_len,
                                                                    args.d_model, args.n_heads, args.e_layers, args.gpt_layers, 
                                                                    args.d_ff, args.embed, iter))
        if not os.path.exists(path):
            os.makedirs(path)
        for epoch in range(self.args.train_epochs):
            time_now = time.time()
            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            progress_bar =  tqdm(enumerate(self.train_loader),total = len(self.train_loader), desc="Processing")
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_trend, seq_seasonal, seq_resid) in progress_bar:

                iter_count += 1
                self.model_optim.zero_grad()
                batch_x = batch_x.float().cuda()

                batch_y = batch_y.float().cuda()
                batch_x_mark = batch_x_mark.float().cuda()
                batch_y_mark = batch_y_mark.float().cuda()

                seq_trend = seq_trend.float().cuda()
                seq_seasonal = seq_seasonal.float().cuda()
                seq_resid = seq_resid.float().cuda()

                outputs, loss_local,clip_loss,STL_loss = self.model(batch_x, iter, seq_trend, seq_seasonal, seq_resid) #+ model(seq_seasonal, ii) + model(seq_resid, ii)
                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :].to(self.device)
                loss = self.criterion(outputs, batch_y) 
                loss = 0.1 * torch.logsumexp(torch.stack([loss,clip_loss.mean(),STL_loss.mean()])/0.1,0) # STCH ,
                train_loss.append(loss.item())
                
                progress_bar.set_description(f"loss: {loss.item():.4f}")
                if (i + 1) % 1000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                self.model_optim.step()
            
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)
            vali_loss = vali(self.model.module, self.vali_data, self.vali_loader, self.criterion, args, self.device, iter)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            if args.cos:
                self.scheduler.step()
                print("lr = {:.10f}".format(self.model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(self.model_optim, epoch + 1, args)
            self.early_stopping(vali_loss, self.model, path)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
    
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path), strict=False)
        print("------------------------------------")
        mse, mae = test(self.model, self.test_data, self.test_loader, args, self.device, iter)
        torch.cuda.empty_cache()
        print('test on the ' + str(args.datasets) + ' dataset: mse:' + str(mse) + ' mae:' + str(mae))
        return mse,mae

    