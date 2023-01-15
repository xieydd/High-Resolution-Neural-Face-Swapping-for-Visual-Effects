import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import our_prune


def main(config, prune_per = 0, prune_type = 'random'):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=config['data_loader']['args']['batch_size'], #512,
        #batch_size=2,
        shuffle=False,
        validation_split=0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)
   
    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    ########### PRUNING ########################################
    amount_ = prune_per
    linear_amount = prune_per
    for name, module in model.named_modules():
    #for module in parameters_to_prune:
        if amount_ == 0:
    	    continue   
        if isinstance(module, torch.nn.Conv2d):
            if prune_type == 'random':
                prune.random_structured(module = module, name = 'weight', amount = amount_, dim=1)
            elif prune_type == 'lastN':
                our_prune.lastN_unstructured(module = module, name = 'weight', amount = amount_)
            else:
                prune.l1_unstructured(module = module, name = 'weight', amount = amount_)
            
    ###### UNCOMMENT TO INCLUDE LINYAR LAYERS ##############
        if isinstance(module, torch.nn.Linear) and name != 'output':
            if prune_type == 'random':
                prune.random_structured(module = module, name = 'weight', amount = linear_amount, dim=1)
            elif prune_type == 'lastN':
                our_prune.lastN_unstructured(module = module, name = 'weight', amount = amount_)
            else:
                prune.l1_unstructured(module = module, name = 'weight', amount = linear_amount)
    ############################################################################
    if amount_ != 0:
    	print('PRUNED '+ str(amount_))
    ###########################################################################

    model.eval()
    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-p', '--prune_per', default=None, type=float,
                      help='add when using pruning (default: None)')
    args.add_argument('-t', '--prune_type', default=None, type=str,
                      help='l1 for l1, random for random and lastN for leacage memory')

    prs_args = args.parse_args()
    config = ConfigParser.from_args(args)
    if prs_args.prune_per is not None:
        if prs_args.prune_type is not None:
            main(config, prs_args.prune_per, prs_args.prune_type)
        else:
            main(config, prs_args.prune_per)
    else:
        main(config)
