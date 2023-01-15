import torch
from sklearn.metrics import multilabel_confusion_matrix

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=5): ####### k=3
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
'''
def TPR(output, target):
    with torch.no_grad():
        predicted =[]
        ground_truth = []
        pred = torch.argmax(output, dim=1)
        predicted.append(pred.cpu().numpy())
        ground_truth.append(target.item())
        assert pred.shape[0] == len(target)

        mcm = multilabel_confusion_matrix(ground_truth, predicted , labels =list(range(10)))
        tn = mcm[:, 0, 0]
        tp = mcm[:, 1, 1]
        fn = mcm[:, 1, 0]
        fp = mcm[:, 0, 1]
    return tp / (tp + fn)



def TNR(output, target):
    with torch.no_grad():
        predicted =[]
        ground_truth = []
        pred = torch.argmax(output, dim=1)
        predicted.append(pred.cpu().numpy())
        ground_truth.append(target.item())
        assert pred.shape[0] == len(target)

        mcm = multilabel_confusion_matrix(ground_truth, predicted , labels =list(range(10)))
        tn = mcm[:, 0, 0]
        tp = mcm[:, 1, 1]
        fn = mcm[:, 1, 0]
        fp = mcm[:, 0, 1]
    return tn / (tn + fp)
'''



def TPR(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)

        class_num = len(output[1])
        conf_matrix = torch.zeros(class_num, class_num)
        for t, p in zip(target, pred):
            conf_matrix[t, p] += 1

        TP = conf_matrix.diag().sum()
        for c in range(class_num):
            idx = torch.ones(class_num).byte()
            idx[c] = 0
            # all class samples not classified as class
            FN = conf_matrix[c, idx].sum()

    return TP / (TP+FN)


def TNR(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)

        class_num = len(output[1])
        conf_matrix = torch.zeros(class_num, class_num)
        for t, p in zip(target, pred):
            conf_matrix[t, p] += 1

        for c in range(class_num):
            idx = torch.ones(class_num).byte()
            idx[c] = 0
            TN = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum()
            FP = conf_matrix[idx, c].sum()

    return TN / (TN+FP)
    

