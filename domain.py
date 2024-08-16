import numpy as np
import torch

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,6,7' # for debug

import warnings
import numpy as np

import argparse
import random

from omegaconf import OmegaConf
from exp_main import Exp_Main as EXP 

def get_init_config(config_path=None):
    config = OmegaConf.load(config_path)
    return config

warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='STEM_TEMPO')

parser.add_argument('--model_id', type=str, default='weather_STEM_TEMPO_6_prompt_learn_336_96_100')
parser.add_argument('--checkpoints', type=str, default='./lora_revin_6domain_checkpoints_1/')
parser.add_argument('--task_name', type=str, default='long_term_forecast')


parser.add_argument('--prompt', type=int, default=1)
parser.add_argument('--num_nodes', type=int, default=1)


parser.add_argument('--seq_len', type=int, default=336)
parser.add_argument('--pred_len', type=int, default=336)#96 192 336 720 
parser.add_argument('--label_len', type=int, default=168)

parser.add_argument('--decay_fac', type=float, default=0.5)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--lradj', type=str, default='type3') # for what
parser.add_argument('--patience', type=int, default=5)

parser.add_argument('--gpt_layers', type=int, default=6)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--d_ff', type=int, default=768)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--enc_in', type=int, default=7)
parser.add_argument('--c_out', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--kernel_size', type=int, default=25)

parser.add_argument('--loss_func', type=str, default='mse')
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--model', type=str, default='STEM_TEMPO')
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--tmax', type=int, default=20)

parser.add_argument('--itr', type=int, default=3)
parser.add_argument('--cos', type=int, default=1)
parser.add_argument('--equal', type=int, default=1, help='1: equal sampling, 0: dont do the equal sampling')
parser.add_argument('--pool', type=bool,default=False, help='whether use prompt pool')
parser.add_argument('--no_stl_loss', action='store_true', help='whether use prompt pool')

parser.add_argument('--stl_weight', type=float, default=0.001)
parser.add_argument('--config_path', type=str, default='./configs/multiple_datasets.yml')
parser.add_argument('--datasets', type=str, default='ETTm1')

parser.add_argument('--use_token', type=int, default=0)
parser.add_argument('--electri_multiplier', type=int, default=1)
parser.add_argument('--traffic_multiplier', type=int, default=1)
parser.add_argument('--embed', type=str, default='timeF')

#args = parser.parse_args([])
args = parser.parse_args()
config = get_init_config(args.config_path)

args.itr = 1

print(args)

SEASONALITY_MAP = {
   "minutely": 1440,
   "10_minutes": 144,
   "half_hourly": 48,
   "hourly": 24,
   "daily": 7,
   "weekly": 1,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}


mses = []
maes = []
exp =  EXP(args,config)
for ii in range(args.itr):
    mse,mae = exp.train_test()

    mses.append(mse)
    maes.append(mae)
print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))

#     mses_s.append(mse_s)
#     maes_s.append(mae_s)
#     mses_t.append(mse_t)
#     maes_t.append(mae_t)
#     mses_f.append(mse_f)
#     maes_f.append(mae_f)
#     mses_5.append(mse_5)
#     maes_5.append(mae_5)
#     mses_6.append(mse_6)
#     maes_6.append(mae_6)
#     mses_7.append(mse_7)
#     maes_7.append(mae_7)
    


# mses = np.array(mses)
# maes = np.array(maes)
# mses_s = np.array(mses_s)
# maes_s = np.array(maes_s)
# mses_t = np.array(mses_t)
# maes_t = np.array(maes_t)
# mses_f = np.array(mses_f)
# maes_f = np.array(maes_f)
# mses_5 = np.array(mses_5)
# maes_5 = np.array(maes_5)
# mses_6 = np.array(mses_6)
# maes_6 = np.array(maes_6)
# mses_7 = np.array(mses_7)
# maes_7 = np.array(maes_7)
# # names = #['weather', 'weather_s', 'weather_t', 'weather_f', 'weather_5', 'ettm2', 'traffic']
# # names = [args.data_name, args.data_name_s, args.data_name_t, args.data_name_f, args.data_name_5, args.data_name_6, args.data_name_7]

# print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
# print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))
# print("mse_s_mean = {:.4f}, mse_s_std = {:.4f}".format(np.mean(mses_s), np.std(mses_s)))
# print("mae_s_mean = {:.4f}, mae_s_std = {:.4f}".format(np.mean(maes_s), np.std(maes_s)))
# print("mse_t_mean = {:.4f}, mse_t_std = {:.4f}".format(np.mean(mses_t), np.std(mses_t)))
# print("mae_t_mean = {:.4f}, mae_t_std = {:.4f}".format(np.mean(maes_t), np.std(maes_t)))
# print("mse_f_mean = {:.4f}, mse_f_std = {:.4f}".format(np.mean(mses_f), np.std(mses_f)))
# print("mae_f_mean = {:.4f}, mae_f_std = {:.4f}".format(np.mean(maes_f), np.std(maes_f)))
# print("mse_5_mean = {:.4f}, mse_5_std = {:.4f}".format(np.mean(mses_5), np.std(mses_5)))
# print("mae_5_mean = {:.4f}, mae_5_std = {:.4f}".format(np.mean(maes_5), np.std(maes_5)))
# print("mse_6_mean = {:.4f}, mse_6_std = {:.4f}".format(np.mean(mses_6), np.std(mses_6)))
# print("mae_6_mean = {:.4f}, mae_6_std = {:.4f}".format(np.mean(maes_6), np.std(maes_6)))
# print("mse_7_mean = {:.4f}, mse_7_std = {:.4f}".format(np.mean(mses_7), np.std(mses_7)))
# print("mae_7_mean = {:.4f}, mae_7_std = {:.4f}".format(np.mean(maes_7), np.std(maes_7)))

# import pandas as pd
# import numpy as np

# # # Create a DataFrame
# # data = {
# #     'Metric': ['MSE', 'MAE'] * 7,
# #     'Mean': [
# #         np.mean(mses), np.mean(maes),
# #         np.mean(mses_s), np.mean(maes_s),
# #         np.mean(mses_t), np.mean(maes_t),
# #         np.mean(mses_f), np.mean(maes_f),
# #         np.mean(mses_5), np.mean(maes_5),
# #         np.mean(mses_6), np.mean(maes_6),
# #         np.mean(mses_7), np.mean(maes_7)
# #     ],
# #     'Standard Deviation': [
# #         np.std(mses), np.std(maes),
# #         np.std(mses_s), np.std(maes_s),
# #         np.std(mses_t), np.std(maes_t),
# #         np.std(mses_f), np.std(maes_f),
# #         np.std(mses_5), np.std(maes_5),
# #         np.std(mses_6), np.std(maes_6),
# #         np.std(mses_7), np.std(maes_7)
# #     ],
# #     'Model': ['weather', 'weather', 'weather_s', 'weather_s', 'weather_t', 'weather_t',
# #               'weather_f', 'weather_f', 'weather_5', 'weather_5', 'ettm2', 'ettm2', 'traffic', 'traffic']
# # }

# # df = pd.DataFrame(data)

# # # Group by the 'Model' column to make the LaTeX table clearer
# # grouped = df.groupby('Model')

# # # Output the DataFrame to a LaTeX table
# # latex_table = grouped.apply(lambda x: x[['Metric', 'Mean', 'Standard Deviation']].to_latex(index=False, float_format="%.4f"))

# # # Print the LaTeX table
# # print(latex_table)


# # LaTeX table header
# latex_table = """
# \\begin{table}[ht]
# \\centering
# \\begin{tabular}{lrr}
# \\toprule
# Model & MSE (Mean ± Std) & MAE (Mean ± Std) \\\\
# \\midrule
# """

# # Collecting data and creating table rows
# metrics = [(mses, maes), (mses_s, maes_s), (mses_t, maes_t), (mses_f, maes_f), (mses_5, maes_5), (mses_6, maes_6), (mses_7, maes_7)]
# for name, (mse_values, mae_values) in zip(names, metrics):
#     mse_mean = np.mean(mse_values)
#     mse_std = np.std(mse_values)
#     mae_mean = np.mean(mae_values)
#     mae_std = np.std(mae_values)
#     latex_table += "{} & {:.4f} ± {:.4f} & {:.4f} ± {:.4f} \\\\\n".format(name, mse_mean, mse_std, mae_mean, mae_std)

# # LaTeX table footer
# latex_table += """
# \\bottomrule
# \\end{tabular}
# \\caption{Summary of model performance.}
# \\label{tab:model_performance}
# \\end{table}
# """

# print(latex_table)


# # Create a DataFrame for the data
# data = {
#     'Model': names,
#     'MSE Mean': [np.mean(mses), np.mean(mses_s), np.mean(mses_t), np.mean(mses_f), np.mean(mses_5), np.mean(mses_6), np.mean(mses_7)],
#     'MSE Std': [np.std(mses), np.std(mses_s), np.std(mses_t), np.std(mses_f), np.std(mses_5), np.std(mses_6), np.std(mses_7)],
#     'MAE Mean': [np.mean(maes), np.mean(maes_s), np.mean(maes_t), np.mean(maes_f), np.mean(maes_5), np.mean(maes_6), np.mean(maes_7)],
#     'MAE Std': [np.std(maes), np.std(maes_s), np.std(maes_t), np.std(maes_f), np.std(maes_5), np.std(maes_6), np.std(maes_7)]
# }

# df = pd.DataFrame(data)

# print(df)
# # Write the DataFrame to an Excel file
# excel_file_path = os.path.join(args.checkpoints, args.model_id + '.xlsx')
# with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
#     df.to_excel(writer, index=False, sheet_name='Performance')

# print(f"Data has been written to {excel_file_path}")
