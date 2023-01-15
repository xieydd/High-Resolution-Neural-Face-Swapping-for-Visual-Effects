import csv
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import funcs


net_list = ['VGG16', 'InceptionV3', 'Resnet50', 'EfficientNetB7']
data_list = ['cifar10', 'cifar100']
prune_type = ['l1', 'random', 'lastN']


for path in pathlib.Path("./prune_results_final").iterdir(): #go throw every csv file in this directory
    if path.is_file():
        with open(path, 'r') as file:

            percent = []
            acc = []
            acc_err =[]

            configuration_name = str(path).split('/')[1].split('.')[0]
            data_name = configuration_name.split('_')[1]

            csvreader = csv.reader(file, delimiter = ' ')

            for row in csvreader: # read csv data
                percent.append(float(row[0].split('%')[0]))
                acc.append(float(row[2]))
                acc_err.append(float(row[7]))


            acc_np = np.array(acc)

            ###### mean ######
            window_size = 5
            acc.insert(0, acc[0])
            acc.insert(0, acc[0])
            acc.append(acc[-1])
            acc.append(acc[-1])
            final_acc = pd.Series(acc).rolling(window_size).mean().tolist()[window_size - 1:]
            #final_acc = acc


            ##### failure rate calculation #######
            ln_acc = [np.log(x) for x in final_acc]
            div_ln_acc = np.gradient(ln_acc)
            neg_ln_div = [-1*x for x in div_ln_acc]


            ###### cut failure rate ##############
            np_neg_ln_div = np.array(neg_ln_div)
            cut_neg_ln_div = np_neg_ln_div[acc_np > funcs.cut_percent(acc[0], data_name)]


            per_np = np.array(percent)
            per_np_cut = per_np[acc_np > funcs.cut_percent(acc[0], data_name)]

            per_to_cut = 0
            cut_per_w = per_np_cut[per_np_cut >= per_to_cut]
            cut_per_g = per_np_cut[(per_np_cut >= per_to_cut)]

            fr_np = np.array(cut_neg_ln_div)
            cut_fr_w = fr_np[per_np_cut >= per_to_cut]
            cut_fr_g = fr_np[(per_np_cut >= per_to_cut)]

            ################# fit ###################
            popt_w, pcov_w = curve_fit(funcs.weib, cut_per_w, cut_fr_w)
            popt_g, pcov_g = curve_fit(funcs.gomp, cut_per_g, cut_fr_g)

            font = { 'weight' : 'bold', 'size'   : 14} #change font of plots
            plt.rc('font', **font)

            plt.figure()
            fig = plt.gcf()
            fig.canvas.set_window_title(configuration_name) #creat the name of the configuration throw the file name

            np_mean = np.array(cut_neg_ln_div)
            np_std = np.array(acc_err)

            fit_per_w = np.linspace(cut_per_w[0], cut_per_w[-1], 100)
            fit_per_g = np.linspace(cut_per_g[0], cut_per_g[-1], 100)

            #accuracy plot
            plt.plot(per_np_cut, cut_neg_ln_div, linewidth=3.0, marker='o', label='test results')
            plt.plot(fit_per_w, funcs.weib(fit_per_w, *popt_w), 'r-', label='weibull fit: a=%.2e, b=%5.3f' % tuple(popt_w))
            #plt.plot(fit_per_g, funcs.weib(fit_per_g, *popt_g), 'g-', label='weibull fit: a=%.2e, b=%5.3f' % tuple(popt_g))
            plt.plot(fit_per_g, funcs.gomp(fit_per_g, *popt_g), 'g-', label='gompertz fit: R=%5.3f, a=%5.3f' % tuple(popt_g))
            plt.title(configuration_name, fontweight="bold", size=20)
            plt.xlabel('prune percent [%]', fontsize = 18)
            plt.ylabel('failure rate', fontsize = 18)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.yscale('log')

    plt.legend()
plt.show()


