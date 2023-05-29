"""
    Base dataset metainfo class.
    based by https://github.com/osmr/imgclsmob/blob/master/pytorch/datasets/cifar10_cls_dataset.py 
    Add unsupervised 
"""

import os


class DatasetMetaInfo(object):
    """
    Base descriptor of dataset.
    """

    def __init__(self):
        self.use_imgrec = False
        self.label = None
        self.root_dir_name = None
        self.root_dir_path = None
        self.dataset_class = None
        self.dataset_class_extra_kwargs = None
        self.num_training_samples = None
        self.in_channels = None
        self.num_classes = None
        self.input_image_size = None
        self.labeled = None
        #train_labeled
        self.train_labeled_metric_capts = None
        self.train_labeled_metric_names = None
        self.train_labeled_metric_extra_kwargs = None
        self.train_labeled_use_weighted_sampler = False
        #train_unlabeled
        self.train_unlabeled_metric_capts = None
        self.train_unlabeled_metric_names = None
        self.train_unlabeled_metric_extra_kwargs = None
        self.train_unlabeled_use_weighted_sampler = False
        #val
        self.val_metric_capts = None
        self.val_metric_names = None
        self.val_metric_extra_kwargs = None
        #test
        self.test_metric_capts = None
        self.test_metric_names = None
        self.test_metric_extra_kwargs = None
        self.saver_acc_ind = None
        self.ml_type = None
        self.allow_hybridize = True
        self.train_net_extra_kwargs = None
        self.test_net_extra_kwargs = None
        self.load_ignore_extra = False

    def add_dataset_parser_arguments(self,
                                     parser,
                                     work_dir_path):
        """
        Create python script parameters (for dataset specific metainfo).

        Parameters:
        ----------
        parser : ArgumentParser
            ArgumentParser instance.
        work_dir_path : str
            Path to working directory.
        """
        parser.add_argument(
            "--data-dir",
            type=str,
            default=os.path.join(work_dir_path, self.root_dir_name),
            help="path to directory with {} dataset".format(self.label))
        parser.add_argument(
            "--num-classes",
            type=int,
            default=self.num_classes,
            help="number of classes")
        parser.add_argument(
            "--in-channels",
            type=int,
            default=self.in_channels,
            help="number of input channels")
        parser.add_argument(
            "--labeled",
            type=int or str,
            choices=[0, 250, 500, 1000, 2000, 4000, 'all'],
            default=self.labeled,
            help="number of labeled samples, you can choose from [0, 250, 500, 1000, 2000, 4000, 'all']")

    def update(self,
               args):
        """
        Update dataset metainfo after user customizing.

        Parameters:
        ----------
        args : ArgumentParser
            Main script arguments.
        """
        self.root_dir_path = args.data_dir
        self.num_classes = args.num_classes
        self.in_channels = args.in_channels
        self.labeled = args.labeled

    def update_from_dataset(self,
                            dataset):
        """
        Update dataset metainfo after a dataset class instance creation.

        Parameters:
        ----------
        args : obj
            A dataset class instance.
        """
        pass