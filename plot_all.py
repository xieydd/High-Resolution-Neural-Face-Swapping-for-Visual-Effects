import csv
import pathlib
import matplotlib.pyplot as plt
import numpy as np

net_list = ['VGG16', 'InceptionV3', 'Resnet50', 'EfficientNetB7']
data_list = ['cifar10', 'cifar100']
prune_type = ['l1', 'random', 'lastN']
directory_names = ['cifar10_lastN', 'cifar100_lastN', 'cifar10_random', 'cifar10_l1', 'cifar100_random', 'cifar100_l1']

for dir in directory_names:

    fig, axis = plt.subplots(2, 2)  # creat subpolt for all plots
    fig = plt.gcf()

    title = dir
    fig.canvas.set_window_title(title)  # creat the name of the configuration throw the file name
    fig.suptitle(title, fontsize='24', fontweight='bold')

    dir_split = dir.split('_')

    for path in pathlib.Path("./prune_results_final/").iterdir():  # go throw every csv file in this directory
        if path.is_file():
            if set(dir_split).issubset(str(path).split('/')[1].split('.')[0].split('_')):
                with open(path, 'r') as file:

                    percent = []
                    loss = []
                    acc = []
                    top5 = []
                    TPR = []
                    TNR = []

                    acc_err = []
                    top5_err = []
                    TPR_err = []
                    TNR_err = []

                    csvreader = csv.reader(file, delimiter=' ')

                    for row in csvreader:  # read csv data
                        percent.append(float(row[0].split('%')[0]))
                        loss.append(float(row[1]))
                        acc.append(float(row[2]))
                        top5.append(float(row[3]))
                        TPR.append(float(row[4]))
                        TNR.append(float(row[5]))
                        acc_err.append(float(row[7]))
                        top5_err.append(float(row[8]))
                        TPR_err.append(float(row[9]))
                        TNR_err.append(float(row[10]))

                    plt.rcParams['font.size'] = '20'

                    # set label according to file name
                    lable = ''
                    if dir_split[0] in net_list:
                        if dir_split[1] in prune_type:
                            lable = str(path).split('/')[1].split('_')[1]
                        else:
                            lable = str(path).split('/')[1].split('.')[0].split('_')[2]
                    elif dir_split[0] in data_list:
                        lable = str(path).split('/')[1].split('_')[0]

                    # set plot color according to net name
                    fmt = ''
                    if lable == net_list[0]:
                        fmt = '#1f77b4'
                    elif lable == net_list[1]:
                        fmt = '#ff7f0e'
                    elif lable == net_list[2]:
                        fmt = '#2ca02e'
                    elif lable == net_list[3]:
                        fmt = '#d62728'

                    # accuracy plot
                    for l in (axis[0, 0].get_xticklabels() + axis[0, 0].get_yticklabels()):
                        l.set_fontsize(20)
                    axis[0, 0].plot(percent, acc, fmt, label=lable, linewidth=3.0, marker='o')
                    axis[0, 0].fill_between(percent, np.array(acc) - np.array(acc_err),
                                            np.array(acc) + np.array(acc_err), facecolor=fmt, alpha=0.2)
                    axis[0, 0].set_xlabel('prune percent [%]', fontsize=22)
                    axis[0, 0].set_ylabel('accuarcy', fontsize=22)

                    # top 5 plot
                    for l in (axis[1, 0].get_xticklabels() + axis[1, 0].get_yticklabels()):
                        l.set_fontsize(20)
                    axis[1, 0].plot(percent, top5, fmt, label=lable, linewidth=3.0, marker='o')
                    axis[1, 0].fill_between(percent, np.array(top5) - np.array(top5_err),
                                            np.array(top5) + np.array(top5_err), facecolor=fmt, alpha=0.2)
                    axis[1, 0].set_xlabel('prune percent [%]', fontsize=22)
                    axis[1, 0].set_ylabel('top 5', fontsize=22)

                    # TPR plot
                    for l in (axis[0, 1].get_xticklabels() + axis[0, 1].get_yticklabels()):
                        l.set_fontsize(20)
                    axis[0, 1].plot(percent, TPR, fmt, label=lable, linewidth=3.0, marker='o')
                    axis[0, 1].fill_between(percent, np.array(TPR) - np.array(TPR_err),
                                            np.array(TPR) + np.array(TPR_err), facecolor=fmt, alpha=0.2)
                    axis[0, 1].set_xlabel('prune percent [%]', fontsize=22)
                    axis[0, 1].set_ylabel('TPR', fontsize=22)

                    # TNR plot
                    for l in (axis[1, 1].get_xticklabels() + axis[1, 1].get_yticklabels()):
                        l.set_fontsize(20)
                    axis[1, 1].plot(percent, TNR, fmt, label=lable, linewidth=3.0, marker='o')
                    axis[1, 1].fill_between(percent, np.array(TNR) - np.array(TNR_err),
                                            np.array(TNR) + np.array(TNR_err), facecolor=fmt, alpha=0.2)
                    axis[1, 1].set_xlabel('prune percent [%]', fontsize=22)
                    axis[1, 1].set_ylabel('TNR', fontsize=22)

    plt.legend(loc='lower right')
plt.show()
