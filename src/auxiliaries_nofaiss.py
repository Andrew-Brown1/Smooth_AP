# repo originally forked from https://github.com/Confusezius/Deep-Metric-Learning-Baselines


################## LIBRARIES ##############################
import warnings
warnings.filterwarnings("ignore")

import numpy as np, os, csv, datetime

import torch, torch.nn as nn

from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl

from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform, pdist, cdist


"""============================================================================================================="""
################# ACQUIRE NUMBER OF WEIGHTS #################
def gimme_params(model):
    """
    Provide number of trainable parameters (i.e. those requiring gradient computation) for input network.

    Args:
        model: PyTorch Network
    Returns:
        int, number of parameters.
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


################# SAVE TRAINING PARAMETERS IN NICE STRING #################
def gimme_save_string(opt):
    """
    Taking the set of parameters and convert it to easy-to-read string, which can be stored later.

    Args:
        opt: argparse.Namespace, contains all training-specific parameters.
    Returns:
        string, returns string summary of parameters.
    """
    varx = vars(opt)
    base_str = ''
    for key in varx:
        base_str += str(key)
        if isinstance(varx[key],dict):
            for sub_key, sub_item in varx[key].items():
                base_str += '\n\t'+str(sub_key)+': '+str(sub_item)
        else:
            base_str += '\n\t'+str(varx[key])
        base_str+='\n\n'
    return base_str



def f1_score(model_generated_cluster_labels, target_labels, feature_coll, computed_centroids):
    """
    NOTE: MOSTLY ADAPTED FROM https://github.com/wzzheng/HDML on Hardness-Aware Deep Metric Learning.

    Args:
        model_generated_cluster_labels: np.ndarray [n_samples x 1], Cluster labels computed on top of data embeddings.
        target_labels:                  np.ndarray [n_samples x 1], ground truth labels for each data sample.
        feature_coll:                   np.ndarray [n_samples x embed_dim], total data embedding made by network.
        computed_centroids:             np.ndarray [num_cluster=num_classes x embed_dim], cluster coordinates
    Returns:
        float, F1-score
    """
    from scipy.special import comb

    d = np.zeros(len(feature_coll))
    for i in range(len(feature_coll)):
        d[i] = np.linalg.norm(feature_coll[i,:] - computed_centroids[model_generated_cluster_labels[i],:])

    labels_pred = np.zeros(len(feature_coll))
    for i in np.unique(model_generated_cluster_labels):
        index = np.where(model_generated_cluster_labels == i)[0]
        ind = np.argmin(d[index])
        cid = index[ind]
        labels_pred[index] = cid


    N = len(target_labels)

    #Cluster n_labels
    avail_labels = np.unique(target_labels)
    n_labels     = len(avail_labels)

    #Count the number of objects in each cluster
    count_cluster = np.zeros(n_labels)
    for i in range(n_labels):
        count_cluster[i] = len(np.where(target_labels == avail_labels[i])[0])

    #Build a mapping from item_id to item index
    keys     = np.unique(labels_pred)
    num_item = len(keys)
    values   = range(num_item)
    item_map = dict()
    for i in range(len(keys)):
        item_map.update([(keys[i], values[i])])


    #Count the number of objects of each item
    count_item = np.zeros(num_item)
    for i in range(N):
        index = item_map[labels_pred[i]]
        count_item[index] = count_item[index] + 1

    #Compute True Positive (TP) plus False Positive (FP) count
    tp_fp = 0
    for k in range(n_labels):
        if count_cluster[k] > 1:
            tp_fp = tp_fp + comb(count_cluster[k], 2)

    #Compute True Positive (TP) count
    tp = 0
    for k in range(n_labels):
        member = np.where(target_labels == avail_labels[k])[0]
        member_ids = labels_pred[member]

        count = np.zeros(num_item)
        for j in range(len(member)):
            index = item_map[member_ids[j]]
            count[index] = count[index] + 1

        for i in range(num_item):
            if count[i] > 1:
                tp = tp + comb(count[i], 2)

    #Compute  False Positive (FP) count
    fp = tp_fp - tp

    #Compute False Negative (FN) count
    count = 0
    for j in range(num_item):
        if count_item[j] > 1:
            count = count + comb(count_item[j], 2)
    fn = count - tp

    # compute F measure
    beta = 1
    P  = tp / (tp + fp)
    R  = tp / (tp + fn)
    F1 = (beta*beta + 1) * P * R / (beta*beta * P + R)

    return F1




"""============================================================================================================="""
def eval_metrics_one_dataset(model, test_dataloader, device, k_vals, opt):
    """
    Compute evaluation metrics on test-dataset, e.g. NMI, F1 and Recall @ k.

    Args:
        model:              PyTorch network, network to compute evaluation metrics for.
        test_dataloader:    PyTorch Dataloader, dataloader for test dataset, should have no shuffling and correct processing.
        device:             torch.device, Device to run inference on.
        k_vals:             list of int, Recall values to compute
        opt:                argparse.Namespace, contains all training-specific parameters.
    Returns:
        F1 score (float), NMI score (float), recall_at_k (list of float), data embedding (np.ndarray)
    """
    torch.cuda.empty_cache()

    _ = model.eval()
    n_classes = len(test_dataloader.dataset.avail_classes)

    with torch.no_grad():
        ### For all test images, extract features
        target_labels, feature_coll = [],[]
        final_iter = tqdm(test_dataloader, desc='Computing Evaluation Metrics...')
        image_paths= [x[0] for x in test_dataloader.dataset.image_list]
        for idx,inp in enumerate(final_iter):
            input_img,target = inp[-1], inp[0]
            target_labels.extend(target.numpy().tolist())
            out = model(input_img.to(device))
            feature_coll.extend(out.cpu().detach().numpy().tolist())

        target_labels = np.hstack(target_labels).reshape(-1,1)
        feature_coll  = np.vstack(feature_coll).astype('float32')

        torch.cuda.empty_cache()

        ### Set Faiss CPU Cluster index
        # cpu_cluster_index = faiss.IndexFlatL2(feature_coll.shape[-1])
        # kmeans            = faiss.Clustering(feature_coll.shape[-1], n_classes)
        # kmeans.niter = 20
        # kmeans.min_points_per_centroid = 1
        # kmeans.max_points_per_centroid = 1000000000

        ### Train Kmeans
        # kmeans.train(feature_coll, cpu_cluster_index)
        # computed_centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes, feature_coll.shape[-1])

        ### Assign feature points to clusters
        # faiss_search_index = faiss.IndexFlatL2(computed_centroids.shape[-1])
        # faiss_search_index.add(computed_centroids)
        # _, model_generated_cluster_labels = faiss_search_index.search(feature_coll, 1)

        kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(feature_coll)
        model_generated_cluster_labels = kmeans.labels_
        computed_centroids = kmeans.cluster_centers_

        ### Compute NMI
        NMI = metrics.cluster.normalized_mutual_info_score(model_generated_cluster_labels.reshape(-1), target_labels.reshape(-1))


        ### Recover max(k_vals) nearest neighbours to use for recall computation
        # faiss_search_index  = faiss.IndexFlatL2(feature_coll.shape[-1])
        # faiss_search_index.add(feature_coll)
        # _, k_closest_points = faiss_search_index.search(feature_coll, int(np.max(k_vals)+1))

        k_closest_points  = squareform(pdist(feature_coll)).argsort(1)[:, :int(np.max(k_vals)+1)]
        k_closest_classes = target_labels.reshape(-1)[k_closest_points[:, 1:]]

        ### Compute Recall
        recall_all_k = []
        for k in k_vals:
            recall_at_k = np.sum([1 for target, recalled_predictions in zip(target_labels, k_closest_classes) if target in recalled_predictions[:k]])/len(target_labels)
            recall_all_k.append(recall_at_k)

        ### Compute F1 Score
        F1 = f1_score(model_generated_cluster_labels, target_labels, feature_coll, computed_centroids)

    return F1, NMI, recall_all_k, feature_coll



def eval_metrics_query_and_gallery_dataset(model, query_dataloader, gallery_dataloader, device, k_vals, opt):
    """
    Compute evaluation metrics on test-dataset, e.g. NMI, F1 and Recall @ k.

    Args:
        model:               PyTorch network, network to compute evaluation metrics for.
        query_dataloader:    PyTorch Dataloader, dataloader for query dataset, for which nearest neighbours in the gallery dataset are retrieved.
        gallery_dataloader:  PyTorch Dataloader, dataloader for gallery dataset, provides target samples which are to be retrieved in correspondance to the query dataset.
        device:              torch.device, Device to run inference on.
        k_vals:              list of int, Recall values to compute
        opt:                 argparse.Namespace, contains all training-specific parameters.
    Returns:
        F1 score (float), NMI score (float), recall_at_ks (list of float), query data embedding (np.ndarray), gallery data embedding (np.ndarray)
    """
    torch.cuda.empty_cache()

    _ = model.eval()
    n_classes = len(query_dataloader.dataset.avail_classes)

    with torch.no_grad():
        ### For all query test images, extract features
        query_target_labels, query_feature_coll     = [],[]
        query_image_paths   = [x[0] for x in query_dataloader.dataset.image_list]
        query_iter = tqdm(query_dataloader, desc='Extraction Query Features')
        for idx,inp in enumerate(query_iter):
            input_img,target = inp[-1], inp[0]
            query_target_labels.extend(target.numpy().tolist())
            out = model(input_img.to(device))
            query_feature_coll.extend(out.cpu().detach().numpy().tolist())

        ### For all gallery test images, extract features
        gallery_target_labels, gallery_feature_coll = [],[]
        gallery_image_paths = [x[0] for x in gallery_dataloader.dataset.image_list]
        gallery_iter = tqdm(gallery_dataloader, desc='Extraction Gallery Features')
        for idx,inp in enumerate(gallery_iter):
            input_img,target = inp[-1], inp[0]
            gallery_target_labels.extend(target.numpy().tolist())
            out = model(input_img.to(device))
            gallery_feature_coll.extend(out.cpu().detach().numpy().tolist())


        query_target_labels, query_feature_coll     = np.hstack(query_target_labels).reshape(-1,1), np.vstack(query_feature_coll).astype('float32')
        gallery_target_labels, gallery_feature_coll = np.hstack(gallery_target_labels).reshape(-1,1), np.vstack(gallery_feature_coll).astype('float32')

        torch.cuda.empty_cache()

        ### Set CPU Cluster index
        stackset    = np.concatenate([query_feature_coll, gallery_feature_coll],axis=0)
        stacklabels = np.concatenate([query_target_labels, gallery_target_labels],axis=0)
        cpu_cluster_index = faiss.IndexFlatL2(stackset.shape[-1])
        kmeans            = faiss.Clustering(stackset.shape[-1], n_classes)
        kmeans.niter = 20
        kmeans.min_points_per_centroid = 1
        kmeans.max_points_per_centroid = 1000000000

        ### Train Kmeans
        kmeans.train(stackset, cpu_cluster_index)
        computed_centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes, stackset.shape[-1])

        ### Assign feature points to clusters
        faiss_search_index = faiss.IndexFlatL2(computed_centroids.shape[-1])
        faiss_search_index.add(computed_centroids)
        _, model_generated_cluster_labels = faiss_search_index.search(stackset, 1)

        ### Compute NMI
        NMI = metrics.cluster.normalized_mutual_info_score(model_generated_cluster_labels.reshape(-1), stacklabels.reshape(-1))

        ### Recover max(k_vals) nearest neighbours to use for recall computation
        faiss_search_index  = faiss.IndexFlatL2(gallery_feature_coll.shape[-1])
        faiss_search_index.add(gallery_feature_coll)
        _, k_closest_points = faiss_search_index.search(query_feature_coll, int(np.max(k_vals)))
        k_closest_classes   = gallery_target_labels.reshape(-1)[k_closest_points]

        ### Compute Recall
        recall_all_k = []
        for k in k_vals:
            recall_at_k = np.sum([1 for target, recalled_predictions in zip(query_target_labels, k_closest_classes) if target in recalled_predictions[:k]])/len(query_target_labels)
            recall_all_k.append(recall_at_k)
        recall_str = ', '.join('@{0}: {1:.4f}'.format(k,rec) for k,rec in zip(k_vals, recall_all_k))

        ### Compute F1 score
        F1 = f1_score(model_generated_cluster_labels, stacklabels, stackset, computed_centroids)

    return F1, NMI, recall_all_k, query_feature_coll, gallery_feature_coll



"""============================================================================================================="""
####### RECOVER CLOSEST EXAMPLE IMAGES #######
def recover_closest_one_dataset(feature_matrix_all, image_paths, save_path, n_image_samples=10, n_closest=3):
    """
    Provide sample recoveries.

    Args:
        feature_matrix_all: np.ndarray [n_samples x embed_dim], full data embedding of test samples.
        image_paths:        list [n_samples], list of datapaths corresponding to <feature_matrix_all>
        save_path:          str, where to store sample image.
        n_image_samples:    Number of sample recoveries.
        n_closest:          Number of closest recoveries to show.
    Returns:
        Nothing!
    """
    image_paths = np.array([x[0] for x in image_paths])
    sample_idxs = np.random.choice(np.arange(len(feature_matrix_all)), n_image_samples)

    # faiss_search_index = faiss.IndexFlatL2(feature_matrix_all.shape[-1])
    # faiss_search_index.add(feature_matrix_all)
    # _, closest_feature_idxs = faiss_search_index.search(feature_matrix_all, n_closest+1)

    closest_feature_idxs  = squareform(pdist(feature_matrix_all)).argsort(1)[:,:n_closest+1]

    sample_paths = image_paths[closest_feature_idxs][sample_idxs]

    f,axes = plt.subplots(n_image_samples, n_closest+1)
    for i,(ax,plot_path) in enumerate(zip(axes.reshape(-1), sample_paths.reshape(-1))):
        ax.imshow(np.array(Image.open(plot_path)))
        ax.set_xticks([])
        ax.set_yticks([])
        if i%(n_closest+1):
            ax.axvline(x=0, color='g', linewidth=13)
        else:
            ax.axvline(x=0, color='r', linewidth=13)
    f.set_size_inches(10,20)
    f.tight_layout()
    f.savefig(save_path)
    plt.close()


####### RECOVER CLOSEST EXAMPLE IMAGES #######
def recover_closest_inshop(query_feature_matrix_all, gallery_feature_matrix_all, query_image_paths, gallery_image_paths, save_path, n_image_samples=10, n_closest=3):
    """
    Provide sample recoveries.

    Args:
        query_feature_matrix_all:   np.ndarray [n_query_samples x embed_dim], full data embedding of query samples.
        gallery_feature_matrix_all: np.ndarray [n_gallery_samples x embed_dim], full data embedding of gallery samples.
        query_image_paths:          list [n_samples], list of datapaths corresponding to <query_feature_matrix_all>
        gallery_image_paths:        list [n_samples], list of datapaths corresponding to <gallery_feature_matrix_all>
        save_path:          str, where to store sample image.
        n_image_samples:    Number of sample recoveries.
        n_closest:          Number of closest recoveries to show.
    Returns:
        Nothing!
    """
    query_image_paths, gallery_image_paths   = np.array(query_image_paths), np.array(gallery_image_paths)
    sample_idxs = np.random.choice(np.arange(len(query_feature_matrix_all)), n_image_samples)

    faiss_search_index = faiss.IndexFlatL2(gallery_feature_matrix_all.shape[-1])
    faiss_search_index.add(gallery_feature_matrix_all)
    _, closest_feature_idxs = faiss_search_index.search(query_feature_matrix_all, n_closest)

    image_paths  = gallery_image_paths[closest_feature_idxs]
    image_paths  = np.concatenate([query_image_paths.reshape(-1,1), image_paths],axis=-1)

    sample_paths = image_paths[closest_feature_idxs][sample_idxs]

    f,axes = plt.subplots(n_image_samples, n_closest+1)
    for i,(ax,plot_path) in enumerate(zip(axes.reshape(-1), sample_paths.reshape(-1))):
        ax.imshow(np.array(Image.open(plot_path)))
        ax.set_xticks([])
        ax.set_yticks([])
        if i%(n_closest+1):
            ax.axvline(x=0, color='g', linewidth=13)
        else:
            ax.axvline(x=0, color='r', linewidth=13)
    f.set_size_inches(10,20)
    f.tight_layout()
    f.savefig(save_path)
    plt.close()



"""============================================================================================================="""
################## SET NETWORK TRAINING CHECKPOINT #####################
def set_checkpoint(model, opt, progress_saver, savepath):
    """
    Store relevant parameters (model and progress saver, as well as parameter-namespace).
    Can be easily extend for other stuff.

    Args:
        model:          PyTorch network, network whose parameters are to be saved.
        opt:            argparse.Namespace, includes all training-specific parameters
        progress_saver: subclass of LOGGER-class, contains a running memory of all training metrics.
        savepath:       str, where to save checkpoint.
    Returns:
        Nothing!
    """
    torch.save({'state_dict':model.state_dict(), 'opt':opt,
                'progress':progress_saver}, savepath)




"""============================================================================================================="""
################## WRITE TO CSV FILE #####################
class CSV_Writer():
    """
    Class to append newly compute training metrics to a csv file
    for data logging.
    Is used together with the LOGGER class.
    """
    def __init__(self, save_path, columns):
        """
        Args:
            save_path: str, where to store the csv file
            columns:   list of str, name of csv columns under which the resp. metrics are stored.
        Returns:
            Nothing!
        """
        self.save_path = save_path
        self.columns   = columns

        with open(self.save_path, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(self.columns)

    def log(self, inputs):
        """
        log one set of entries to the csv.

        Args:
            inputs: [list of int/str/float], values to append to the csv. Has to be of the same length as self.columns.
        Returns:
            Nothing!
        """
        with open(self.save_path, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(inputs)



################## PLOT SUMMARY IMAGE #####################
class InfoPlotter():
    """
    Plotter class to visualize training progression by showing
    different metrics.
    """
    def __init__(self, save_path, title='Training Log', figsize=(20,15)):
        """
        Args:
            save_path: str, where to store the create plot.
            title:     placeholder title of plot
            figsize:   base size of saved figure
        Returns:
            Nothing!
        """
        self.save_path = save_path
        self.title     = title
        self.figsize   = figsize
        #Colors for validation lines
        self.v_colors    = ['r','g','b','y','m','k','c']
        #Colors for training lines
        self.t_colors    = ['k','b','r','g']

    def make_plot(self, t_epochs, v_epochs, t_metrics, v_metrics, t_labels, v_labels, appendix=None):
        """
        Given a list of iterated epochs, visualize the progression of various training/testing metrics.

        Args:
            t_epochs:  [list of int/float], list of epochs for which training metrics were collected (e.g. Training Loss)
            v_epochs:  [list of int/float], list of epochs for which validation metrics were collected (e.g. Recall @ k)
            t_metrics: [list of float], list of training metrics per epoch
            v_metrics: [list of list of int/float], contains all computed validation metrics
            t_labels, v_labels: [list of str], names for each metric that is plotted.
        Returns:
            Nothing!
        """
        plt.style.use('ggplot')

        f,axes = plt.subplots(1,2)

        #Visualize Training Loss
        for i in range(len(t_metrics)):
            axes[0].plot(t_epochs, t_metrics[i], '-{}'.format(self.t_colors[i]), linewidth=1, label=t_labels[i])
        axes[0].set_title('Training Performance', fontsize=19)

        axes[0].legend(fontsize=16)

        axes[0].tick_params(axis='both', which='major', labelsize=16)
        axes[0].tick_params(axis='both', which='minor', labelsize=16)

        #Visualize Validation metrics
        for i in range(len(v_metrics)):
            axes[1].plot(v_epochs, v_metrics[i], '-{}'.format(self.v_colors[i]), linewidth=1, label=v_labels[i])
        axes[1].set_title(self.title, fontsize=19)

        axes[1].legend(fontsize=16)

        axes[1].tick_params(axis='both', which='major', labelsize=16)
        axes[1].tick_params(axis='both', which='minor', labelsize=16)

        f.set_size_inches(2*self.figsize[0], self.figsize[1])

        savepath = self.save_path
        f.savefig(self.save_path, bbox_inches='tight')

        plt.close()


################## GENERATE LOGGING FOLDER/FILES #######################
def set_logging(opt):
    """
    Generate the folder in which everything is saved.
    If opt.savename is given, folder will take on said name.
    If not, a name based on the start time is provided.
    If the folder already exists, it will by iterated until it can be created without
    deleting existing data.
    The current opt.save_path will be extended to account for the new save_folder name.

    Args:
        opt: argparse.Namespace, contains all training-specific parameters.
    Returns:
        Nothing!
    """
    checkfolder = opt.save_path+'/'+opt.savename

    #Create start-time-based name if opt.savename is not give.
    if opt.savename == '':
        date = datetime.datetime.now()
        time_string = '{}-{}-{}-{}-{}-{}'.format(date.year, date.month, date.day, date.hour, date.minute, date.second)
        checkfolder = opt.save_path+'/{}_{}_'.format(opt.dataset.upper(), opt.arch.upper())+time_string

    #If folder already exists, iterate over it until is doesn't.
    counter     = 1
    while os.path.exists(checkfolder):
        checkfolder = opt.save_path+'/'+opt.savename+'_'+str(counter)
        counter += 1

    #Create Folder
    os.makedirs(checkfolder)
    opt.save_path = checkfolder

    #Store training parameters as text and pickle in said folder.
    with open(opt.save_path+'/Parameter_Info.txt','w') as f:
        f.write(gimme_save_string(opt))
    pkl.dump(opt,open(opt.save_path+"/hypa.pkl","wb"))


class LOGGER():
    """
    This class provides a collection of logging properties that are useful for training.
    These include setting the save folder, in which progression of training/testing metrics is visualized,
    csv log-files are stored, sample recoveries are plotted and an internal data saver.
    """
    def __init__(self, opt, metrics_to_log, name='Basic', start_new=True):
        """
        Args:
            opt:               argparse.Namespace, contains all training-specific parameters.
            metrics_to_log:    dict, dictionary which shows in what structure the data should be saved.
                               is given as the output of aux.metrics_to_examine. Example:
                               {'train': ['Epochs', 'Time', 'Train Loss', 'Time'],
                                'val': ['Epochs','Time','NMI','F1', 'Recall @ 1','Recall @ 2','Recall @ 4','Recall @ 8']}
            name:              Name of this logger. Will be used to distinguish logged files from other LOGGER instances.
            start_new:         If set to true, a new save folder will be created initially.
        Returns:
            Nothing!
        """
        self.prop           = opt
        self.metrics_to_log = metrics_to_log

        ### Make Logging Directories
        if start_new: set_logging(opt)

        ### Set INFO-PLOTS
        if self.prop.dataset != 'vehicle_id':
            self.info_plot = InfoPlotter(opt.save_path+'/InfoPlot_{}.svg'.format(name))
        else:
            self.info_plot = {'Set {}'.format(i): InfoPlotter(opt.save_path+'/InfoPlot_{}_Set{}.svg'.format(name,i+1)) for i in range(3)}

        ### Set Progress Saver Dict
        self.progress_saver = self.provide_progress_saver(metrics_to_log)

        ### Set CSV Writters
        self.csv_loggers= {mode:CSV_Writer(opt.save_path+'/log_'+mode+'_'+name+'.csv', lognames) for mode, lognames in metrics_to_log.items()}


    def provide_progress_saver(self, metrics_to_log):
        """
        Provide Progress Saver dictionary.

        Args:
            metrics_to_log: see __init__(). Describes the structure of Progress_Saver.
        """
        Progress_Saver = {key:{sub_key:[] for sub_key in metrics_to_log[key]} for key in metrics_to_log.keys()}
        return Progress_Saver

    def log(self, main_keys, metric_keys, values):
        """
        Actually log new values in csv and Progress Saver dict internally.
        Args:
            main_keys:      Main key in which data will be stored. Normally is either 'train' for training metrics or 'val' for validation metrics.
            metric_keys:    Needs to follow the list length of self.progress_saver[main_key(s)]. List of metric keys that are extended with new values.
            values:         Needs to be a list of the same structure as metric_keys. Actual values that are appended.
        """
        if not isinstance(main_keys, list):   main_keys = [main_keys]
        if not isinstance(metric_keys, list): metric_keys = [metric_keys]
        if not isinstance(values, list):      values = [values]

        #Log data to progress saver dict.
        for main_key in main_keys:
            for value, metric_key in zip(values, metric_keys):
                self.progress_saver[main_key][metric_key].append(value)

        #Append data to csv.
        self.csv_loggers[main_key].log(values)

    def update_info_plot(self):
        """
        Create a new updated version of training/metric progression plot.

        Args:
            None
        Returns:
            Nothing!
        """
        t_epochs         = self.progress_saver['val']['Epochs']
        t_loss_list      = [self.progress_saver['train']['Train Loss']]
        t_legend_handles = ['Train Loss']

        v_epochs         = self.progress_saver['val']['Epochs']
        #Because Vehicle-ID normally uses three different test sets, a distinction has to be made.
        if self.prop.dataset != 'vehicle_id':
            title = ' | '.join(key+': {0:3.3f}'.format(np.max(item)) for key,item in self.progress_saver['val'].items() if key not in ['Time', 'Epochs'])
            self.info_plot.title = title
            v_metric_list    = [self.progress_saver['val'][key] for key in self.progress_saver['val'].keys() if key not in ['Time', 'Epochs']]
            v_legend_handles = [key for key in self.progress_saver['val'].keys() if key not in ['Time', 'Epochs']]

            self.info_plot.make_plot(t_epochs, v_epochs, t_loss_list, v_metric_list, t_legend_handles, v_legend_handles)
        else:
            #Iterate over all test sets.
            for i in range(3):
                title = ' | '.join(key+': {0:3.3f}'.format(np.max(item)) for key,item in self.progress_saver['val'].items() if key not in ['Time', 'Epochs'] and 'Set {}'.format(i) in key)
                self.info_plot['Set {}'.format(i)].title = title
                v_metric_list    = [self.progress_saver['val'][key] for key in self.progress_saver['val'].keys() if key not in ['Time', 'Epochs'] and 'Set {}'.format(i) in key]
                v_legend_handles = [key for key in self.progress_saver['val'].keys() if key not in ['Time', 'Epochs'] and 'Set {}'.format(i) in key]
                self.info_plot['Set {}'.format(i)].make_plot(t_epochs, v_epochs, t_loss_list, v_metric_list, t_legend_handles, v_legend_handles, appendix='set_{}'.format(i))

def metrics_to_examine(dataset, k_vals):
    """
    Please only use either of the following keys:
    -> Epochs, Time, Train Loss for training
    -> Epochs, Time, NMI, F1 & Recall @ k for validation

    Args:
        dataset: str, dataset for which a storing structure for LOGGER.progress_saver is to be made.
        k_vals:  list of int, Recall @ k - values.
    Returns:
        metric_dict: Dictionary representing the storing structure for LOGGER.progress_saver. See LOGGER.__init__() for an example.
    """
    metric_dict        = {'train':['Epochs','Time','Train Loss']}

    if dataset=='vehicle_id':
        metric_dict['val'] = ['Epochs','Time']
        #Vehicle_ID uses three test sets
        for i in range(3):
            metric_dict['val'] += ['Set {} NMI'.format(i), 'Set {} F1'.format(i)]
            for k in k_vals:
                metric_dict['val'] += ['Set {} Recall @ {}'.format(i,k)]
    else:
        metric_dict['val'] = ['Epochs','Time','NMI', 'F1']
        metric_dict['val'] += ['Recall @ {}'.format(k) for k in k_vals]

    return metric_dict



"""================================================================================================="""
def run_kmeans(features, n_cluster):
    """
    Run kmeans on a set of features to find <n_cluster> cluster.

    Args:
        features:  np.ndarrary [n_samples x embed_dim], embedding training/testing samples for which kmeans should be performed.
        n_cluster: int, number of cluster.
    Returns:
        cluster_assignments: np.ndarray [n_samples x 1], per sample provide the respective cluster label it belongs to.
    """
    n_samples, dim = features.shape
    kmeans = faiss.Kmeans(dim, n_cluster)
    kmeans.n_iter, kmeans.min_points_per_centroid, kmeans.max_points_per_centroid = 20,5,1000000000
    kmeans.train(features)
    _, cluster_assignments = kmeans.index.search(features,1)
    return cluster_assignments



"""============================================================================================================="""
def save_graph(opt, model):
    """
    Generate Network Graph.
    NOTE: Requires the installation of the graphviz library on you system.

    Args:
        opt:   argparse.Namespace, contains all training-specific parameters.
        model: PyTorch Network, network for which the computational graph should be visualized.
    Returns:
        Nothing!
    """
    inp = torch.randn((1,3,224,224)).to(opt.device)
    network_output = model(inp)
    if isinstance(network_output, dict): network_output = network_output['Class']

    from graphviz import Digraph
    def make_dot(var, savename, params=None):
        """
        Generate a symbolic representation of the network graph.
        """
        if params is not None:
            assert all(isinstance(p, Variable) for p in params.values())
            param_map = {id(v): k for k, v in params.items()}

        node_attr = dict(style='filled',
                         shape='box',
                         align='left',
                         fontsize='6',
                         ranksep='0.1',
                         height='0.6',
                         width='1')
        dot  = Digraph(node_attr=node_attr, format='svg', graph_attr=dict(size="40,10", rankdir='LR', rank='same'))
        seen = set()

        def size_to_str(size):
            return '('+(', ').join(['%d' % v for v in size])+')'

        def add_nodes(var):
            replacements  = ['Backward', 'Th', 'Cudnn']
            color_assigns = {'Convolution':'orange',
                             'ConvolutionTranspose': 'lightblue',
                             'Add': 'red',
                             'Cat': 'green',
                             'Softmax': 'yellow',
                             'Sigmoid': 'yellow',
                             'Copys':   'yellow'}
            if var not in seen:
                op1 = torch.is_tensor(var)
                op2 = not torch.is_tensor(var) and str(type(var).__name__)!='AccumulateGrad'

                text = str(type(var).__name__)
                for rep in replacements:
                    text = text.replace(rep, '')
                color = color_assigns[text] if text in color_assigns.keys() else 'gray'

                if 'Pool' in text: color = 'lightblue'

                if op1 or op2:
                    if hasattr(var, 'next_functions'):
                        count = 0
                        for i, u in enumerate(var.next_functions):
                            if str(type(u[0]).__name__)=='AccumulateGrad':
                                if count==0: attr_text = '\nParameter Sizes:\n'
                                attr_text += size_to_str(u[0].variable.size())
                                count += 1
                                attr_text += ' '
                        if count>0: text += attr_text


                if op1:
                    dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
                if op2:
                    dot.node(str(id(var)), text, fillcolor=color)

                seen.add(var)

                if op1 or op2:
                    if hasattr(var, 'next_functions'):
                        for u in var.next_functions:
                            if u[0] is not None:
                                if str(type(u[0]).__name__)!='AccumulateGrad':
                                    dot.edge(str(id(u[0])), str(id(var)))
                                    add_nodes(u[0])
                    if hasattr(var, 'saved_tensors'):
                        for t in var.saved_tensors:
                            dot.edge(str(id(t)), str(id(var)))
                            add_nodes(t)

        add_nodes(var.grad_fn)
        dot.save(savename)
        return dot

    if not os.path.exists(opt.save_path):
        raise Exception('No save folder {} available!'.format(opt.save_path))

    viz_graph = make_dot(network_output, opt.save_path+"/Network_Graphs"+"/{}_network_graph".format(opt.arch))
    viz_graph.format = 'svg'
    viz_graph.render()

    torch.cuda.empty_cache()
