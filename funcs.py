import numpy as np


def gomp(x, R, a):
    return R * np.exp(a * x)


def weib(x, a, b):
    return a * x ** b


def cut10(max_acc, percent):
    return ((max_acc - 0.1) / 100) * percent + 0.1


def cut100(max_acc, percent):
    return ((max_acc - 0.01) / 100) * percent + 0.01


def cut_percent(max_acc, data_name):
    percent = 7
    if data_name == 'cifar10':
        return cut10(max_acc, percent)
    elif data_name == 'cifar100':
        return cut100(max_acc, percent)
