"""
Base dataset class
"""

import os

class DatasetInfo(object):
    def __init__(self):
        self.use_imgrec = None
        self.label = None
        self.root_dir = None
        self.num_labels = None
        self.dataset_class = None
        self.dataset_class_extra_kwargs = None
        self.num_train_samples = None
        self.in_channels = None
        self.num_classes = None
        self.input_size = None
        self.train_metric_capts = None
        self.train_metric_names = None
        self.train_metric_extra_kwargs = None
        self.train_use_weighted_sampler = False
        self.train_un_metric_capts = None
        self.train_un_metric_names = None
        self.train_use_multiepoch_sampler = False
        self.val_metric_capts = None
        self.val_metric_names = None
        self.val_metric_extra_kwargs = None
        self.test_metric_capts = None
        self.test_metric_names = None
        self.test_metric_extra_kwargs = None
        self.saver_acc_ind = None
        self.ml_type = None
        self.allow_hybridize = True
        self.train_net_extra_kwargs = None
        self.test_net_extra_kwargs = None
        self.load_ignore_extra = False

    def add_dataset_parser_agument(self, parser, work_dir):
        """
        create python argparse parser for dataset
        """
        parser.add_argument('--data-dir', type=str, default=os.path.join(work_dir, self.root_dir),
                            help='path to directory with data')
        parser.add_argument('--num-classes', type=int, default=self.num_classes,
                            help='number of classes')
        parser.add_argument('--in-channels', type=int, default=self.in_channels,
                            help='number of input channels')
    
    def update(self, args):
        """
        Update dataset info using parsed args
        """
        self.root_dir = args.data_dir
        self.num_classes = args.num_classes
        self.in_channels = args.in_channels

    def update_from_dataset(self, dataset):
        """
        Update dataset info using dataset class instance
        """
        pass
    