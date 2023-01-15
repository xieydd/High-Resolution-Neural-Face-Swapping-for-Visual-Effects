import csv
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import funcs

net_list = ['VGG16', 'InceptionV3', 'Resnet50', 'EfficientNetB7']
data_list = ['cifar10', 'cifar100']
prune_type = ['l1', 'random', 'lastN']
directory_names = ['cifar10_lastN', 'cifar100_lastN', 'cifar10_random', 'cifar10_l1', 'cifar100_random', 'cifar100_l1']

for dir in directory_names:

    dir_split = dir.split('_')
    plt.figure()
    for path in pathlib.Path("./prune_results_final").iterdir():  # go throw every csv file in this directory
        if path.is_file():
            if set(dir_split).issubset(str(path).split('/')[1].split('.')[0].split('_')):
                with open(path, 'r') as file:

                    percent = []
                    acc = []
                    acc_err = []

                    csvreader = csv.reader(file, delimiter=' ')
                    data_name = str(path).split('/')[1].split('.')[0].split('_')[1]

                    for row in csvreader:  # read csv data
                        percent.append(float(row[0].split('%')[0]))
                        acc.append(float(row[2]))
                        acc_err.append(float(row[7]))

                    acc_np = np.array(acc)
                    acc_err_max = [acc[i] + acc_err[i] for i in range(len(acc))]
                    acc_err_min = [acc[i] - acc_err[i] for i in range(len(acc))]

                    ###### mean ######
                    window_size = 5

                    acc.insert(0, acc[0])
                    acc.insert(0, acc[0])
                    acc.append(acc[-1])
                    acc.append(acc[-1])

                    acc_err_max.insert(0, acc_err_max[0])
                    acc_err_max.insert(0, acc_err_max[0])
                    acc_err_max.append(acc_err_max[-1])
                    acc_err_max.append(acc_err_max[-1])

                    acc_err_min.insert(0, acc_err_min[0])
                    acc_err_min.insert(0, acc_err_min[0])
                    acc_err_min.append(acc_err_min[-1])
                    acc_err_min.append(acc_err_min[-1])

                    # final_acc = acc
                    final_acc = pd.Series(acc).rolling(window_size).mean().tolist()[window_size - 1:]
                    final_acc_max = pd.Series(acc_err_max).rolling(window_size).mean().tolist()[window_size - 1:]
                    final_acc_min = pd.Series(acc_err_min).rolling(window_size).mean().tolist()[window_size - 1:]

                    ##### failure rate calculation #######
                    ln_acc = [np.log(x) for x in final_acc]
                    div_ln_acc = np.gradient(ln_acc)
                    neg_ln_div = [-1 * x for x in div_ln_acc]

                    ln_acc_max = [np.log(x) for x in final_acc_max]
                    div_acc_err_max = np.gradient(ln_acc_max)
                    neg_acc_max = [-1 * x for x in div_acc_err_max]

                    ln_acc_min = [np.log(x) for x in final_acc_min]
                    div_acc_err_min = np.gradient(ln_acc_min)
                    neg_acc_min = [-1 * x for x in div_acc_err_min]

                    neg_ln_div_acc_err = [(neg_acc_max[i] - neg_acc_min[i]) / 2 for i in range(len(neg_acc_max))]

                    ###### cut failure rate ##############
                    np_neg_ln_div = np.array(neg_ln_div)
                    cut_neg_ln_div = np_neg_ln_div[acc_np > funcs.cut_percent(acc[0], data_name)]

                    np_neg_acc_err = np.array(neg_ln_div_acc_err)
                    cut_neg_acc_err = np_neg_acc_err[acc_np > funcs.cut_percent(acc[0], data_name)]

                    per_np = np.array(percent)
                    per_np_cut = per_np[acc_np > funcs.cut_percent(acc[0], data_name)]

                    plt.rcParams['font.size'] = '14'

                    # set lable acording to file name
                    lable = ''
                    if dir_split[0] in net_list:
                        if dir_split[1] in prune_type:
                            lable = str(path).split('/')[1].split('_')[1]
                        else:
                            lable = str(path).split('/')[1].split('.')[0].split('_')[2]
                    elif dir_split[0] in data_list:
                        lable = str(path).split('/')[1].split('_')[0]

                    fmt = ''
                    if lable == net_list[0]:
                        fmt = '#1f77b4'
                    elif lable == net_list[1]:
                        fmt = '#ff7f0e'
                    elif lable == net_list[2]:
                        fmt = '#2ca02e'
                    elif lable == net_list[3]:
                        fmt = '#d62728'

                    np_mean = np.array(cut_neg_ln_div)
                    np_std = np.array(cut_neg_acc_err)

                    # accuracy plot
                    plt.plot(per_np_cut, cut_neg_ln_div, fmt, label=lable, linewidth=3.0, marker='o')
                    plt.fill_between(per_np_cut, np_mean - np_std, np_mean + np_std, facecolor=fmt, alpha=0.2)
                    plt.title(dir, fontweight="bold", size=20)
                    plt.xlabel('prune percent [%]', fontsize=18)
                    plt.ylabel('failure rate', fontsize=18)
                    plt.xticks(fontsize=14)
                    plt.yticks(fontsize=14)
                    plt.yscale('log')

    plt.legend()
plt.show()
