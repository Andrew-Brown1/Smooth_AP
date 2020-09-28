import os, torch, argparse
import netlib as netlib
import auxiliaries as aux
import datasets as data
import evaluate as eval

if __name__ == '__main__':

    ################## INPUT ARGUMENTS ###################
    parser = argparse.ArgumentParser()
    ####### Main Parameter: Dataset to use for Training
    parser.add_argument('--dataset', default='vehicle_id', type=str, help='Dataset to use.',
                        choices=['Inaturalist', 'vehicle_id'])
    parser.add_argument('--source_path', default='/scratch/shared/beegfs/abrown/datasets', type=str,
                        help='Path to training data.')
    parser.add_argument('--save_path', default=os.getcwd() + '/Training_Results', type=str,
                        help='Where to save everything.')
    parser.add_argument('--savename', default='', type=str,
                        help='Save folder name if any special information is to be included.')

    ### General Training Parameters
    parser.add_argument('--kernels', default=8, type=int, help='Number of workers for pytorch dataloader.')
    parser.add_argument('--bs', default=112, type=int, help='Mini-Batchsize to use.')
    parser.add_argument('--samples_per_class', default=4, type=int,help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')
    parser.add_argument('--loss', default='smoothap', type=str)

    ##### Evaluation Settings
    parser.add_argument('--k_vals', nargs='+', default=[1, 2, 4, 8], type=int, help='Recall @ Values.')
    ##### Network parameters
    parser.add_argument('--embed_dim', default=512, type=int,
                        help='Embedding dimensionality of the network. Note: in literature, dim=128 is used for ResNet50 and dim=512 for GoogLeNet.')
    parser.add_argument('--arch', default='resnet50', type=str,
                        help='Network backend choice: resnet50, googlenet, BNinception')
    parser.add_argument('--gpu', default=0, type=int, help='GPU-id for GPU to use.')
    parser.add_argument('--resume', default='', type=str, help='path to where weights to be evaluated are saved.')
    parser.add_argument('--not_pretrained', action='store_true',
                        help='If added, the network will be trained WITHOUT ImageNet-pretrained weights.')
    opt = parser.parse_args()

    """============================================================================"""
    opt.source_path += '/' + opt.dataset

    if opt.dataset == 'Inaturalist':
        opt.n_epochs = 90
        opt.tau = [40, 70]
        opt.k_vals = [1, 4, 16, 32]

    if opt.dataset == 'vehicle_id':
        opt.k_vals = [1, 5]

    metrics_to_log = aux.metrics_to_examine(opt.dataset, opt.k_vals)
    LOG = aux.LOGGER(opt, metrics_to_log, name='Base', start_new=True)

    """============================================================================"""
    ##################### NETWORK SETUP ##################

    opt.device = torch.device('cuda')
    model = netlib.networkselect(opt)

    # Push to Device
    _ = model.to(opt.device)

    """============================================================================"""
    #################### DATALOADER SETUPS ##################
    # Returns a dictionary containing 'training', 'testing', and 'evaluation' dataloaders.
    # The 'testing'-dataloader corresponds to the validation set, and the 'evaluation'-dataloader
    # Is simply using the training set, however running under the same rules as 'testing' dataloader,
    # i.e. no shuffling and no random cropping.
    dataloaders = data.give_dataloaders(opt.dataset, opt)
    # Because the number of supervised classes is dataset dependent, we store them after
    # initializing the dataloader
    opt.num_classes = len(dataloaders['training'].dataset.avail_classes)

    if opt.dataset == 'Inaturalist':
        eval_params = {'dataloader': dataloaders['testing'], 'model': model, 'opt': opt, 'epoch': 0}

    elif opt.dataset == 'vehicle_id':
        eval_params = {
            'dataloaders': [dataloaders['testing_set1'], dataloaders['testing_set2'], dataloaders['testing_set3']],
            'model': model, 'opt': opt, 'epoch': 0}

    """============================================================================"""
    ####################evaluation ##################

    results = eval.evaluate(opt.dataset, LOG, save=True, **eval_params)
