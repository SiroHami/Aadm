"""
util units for training

this code is from osmr's github
https://github.com/osmr/imgclsmob/blob/f2993d3ce73a2f7ddba05da3891defb08547d504/common/env_stats.py#L5
"""

import json
import os
import platform
import subprocess
import sys
import logging

import torch

from models.model_provider import get_model


def accuracy(metric, y_true, y_pred):
    return metric(y_true, y_pred).numpy()

def get_data(Dataset, batch_size, num_workers, is_train=True, **kwargs):
    """
    Get data loader by dataset.

    Parameters:
    ----------
    Dataset : Dataset
        Dataset for loading.
    batch_size : int
        Batch size.
    num_workers : int
        Number of background workers.
    is_train : bool, default True
        Whether to use training subset.

    Returns:
    -------
    DataLoader
        Data loader.
    """
    dataset = Dataset(
        is_train=is_train,
        **kwargs)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers)
    return data_loader

def prepare_model(model_name,
                  use_pretrained,
                  pretrained_model_file_path,
                  use_cuda,
                  use_data_parallel=True,
                  net_extra_kwargs=None,
                  load_ignore_extra=False,
                  num_classes=None,
                  in_channels=None,
                  remap_to_cpu=False,
                  remove_module=False):
    """
    Create and initialize model by name.

    Parameters:
    ----------
    model_name : str
        Model name.
    use_pretrained : bool
        Whether to use pretrained weights.
    pretrained_model_file_path : str
        Path to file with pretrained weights.
    use_cuda : bool
        Whether to use CUDA.
    use_data_parallel : bool, default True
        Whether to use parallelization.
    net_extra_kwargs : dict, default None
        Extra parameters for model.
    load_ignore_extra : bool, default False
        Whether to ignore extra layers in pretrained model.
    num_classes : int, default None
        Number of classes.
    in_channels : int, default None
        Number of input channels.
    remap_to_cpu : bool, default False
        Whether to remape model to CPU during loading.
    remove_module : bool, default False
        Whether to remove module from loaded model.

    Returns:
    -------
    Module
        Model.
    """
    kwargs = {"pretrained": use_pretrained}
    if num_classes is not None:
        kwargs["num_classes"] = num_classes
    if in_channels is not None:
        kwargs["in_channels"] = in_channels
    if net_extra_kwargs is not None:
        kwargs.update(net_extra_kwargs)

    net = get_model(model_name, **kwargs)

    if pretrained_model_file_path:
        assert (os.path.isfile(pretrained_model_file_path))
        logging.info("Loading model: {}".format(pretrained_model_file_path))
        checkpoint = torch.load(
            pretrained_model_file_path,
            map_location=(None if use_cuda and not remap_to_cpu else "cpu"))
        if (type(checkpoint) == dict) and ("state_dict" in checkpoint):
            checkpoint = checkpoint["state_dict"]

        if load_ignore_extra:
            pretrained_state = checkpoint
            model_dict = net.state_dict()
            pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_dict}
            net.load_state_dict(pretrained_state)
        else:
            if remove_module:
                net_tmp = torch.nn.DataParallel(net)
                net_tmp.load_state_dict(checkpoint)
                net.load_state_dict(net_tmp.module.cpu().state_dict())
            else:
                net.load_state_dict(checkpoint)

    if use_data_parallel and use_cuda:
        net = torch.nn.DataParallel(net)

    if use_cuda:
        net = net.cuda()

    return net

def prepare_logger(logging_dir_path,
                   logging_file_name):
    """
    Prepare logger.

    Parameters:
    ----------
    logging_dir_path : str
        Path to logging directory.
    logging_file_name : str
        Name of logging file.

    Returns:
    -------
    Logger
        Logger instance.
    bool
        If the logging file exist.
    """
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # sh = logging.StreamHandler()
    # logger.addHandler(sh)
    log_file_exist = False
    if logging_dir_path is not None and logging_dir_path:
        log_file_path = os.path.join(logging_dir_path, logging_file_name)
        if not os.path.exists(logging_dir_path):
            os.makedirs(logging_dir_path)
            log_file_exist = False
        else:
            log_file_exist = (os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0)
        fh = logging.FileHandler(log_file_path)
        logger.addHandler(fh)
        if log_file_exist:
            logging.info("--------------------------------")
    return logger, log_file_exist


def initialize_logging(logging_dir_path,
                       logging_file_name,
                       script_args,
                       log_packages,
                       log_pip_packages):
    """
    Initialize logging subsystem.

    Parameters:
    ----------
    logging_dir_path : str
        Path to logging directory.
    logging_file_name : str
        Name of logging file.
    script_args : ArgumentParser
        Main script arguments.
    log_packages : bool
        Whether to log packages info.
    log_pip_packages : bool
        Whether to log pip-packages info.

    Returns:
    -------
    Logger
        Logger instance.
    bool
        If the logging file exist.
    """
    logger, log_file_exist = prepare_logger(
        logging_dir_path=logging_dir_path,
        logging_file_name=logging_file_name)
    logging.info("Script command line:\n{}".format(" ".join(sys.argv)))
    logging.info("Script arguments:\n{}".format(script_args))
    packages = log_packages.replace(" ", "").split(",") if type(log_packages) == str else log_packages
    pip_packages = log_pip_packages.replace(" ", "").split(",") if type(log_pip_packages) == str else log_pip_packages
    if (log_packages is not None) and (log_pip_packages is not None):
        logging.info("Env_stats:\n{}".format(get_env_stats(
            packages=packages,
            pip_packages=pip_packages)))
    return logger, log_file_exist


def get_pip_versions(package_list,
                     python_version=""):
    """
    Get packages information by using 'pip show' command.

    Parameters:
    ----------
    package_list : list of str
        List of package names.
    python_version : str, default ''
        Python version ('2', '3', '') appended to 'pip' command.

    Returns:
    -------
    dict
        Dictionary with module descriptions.
    """
    module_versions = {}
    for module in package_list:
        try:
            out_bytes = subprocess.check_output([
                "pip{0}".format(python_version),
                "show", module])
            out_text = out_bytes.decode("utf-8").strip()
        except (subprocess.CalledProcessError, OSError):
            out_text = None
        module_versions[module] = out_text
    return module_versions


def get_package_versions(package_list):
    """
    Get packages information by inspecting __version__ attribute.

    Parameters:
    ----------
    package_list : list of str
        List of package names.

    Returns:
    -------
    dict
        Dictionary with module descriptions.
    """
    module_versions = {}
    for module in package_list:
        try:
            module_versions[module] = __import__(module).__version__
        except ImportError:
            module_versions[module] = None
        except AttributeError:
            module_versions[module] = "unknown"
    return module_versions


def get_pyenv_info(packages,
                   pip_packages,
                   python_ver,
                   pwd,
                   git,
                   sys_info=True):
    """
    Get all available information about Python environment: packages information, Python version, current path,
    git revision.

    Parameters:
    ----------
    packages : list of str
        list of package names to inspect only __version__.
    pip_packages : list of str
        List of package names to inspect by 'pip show'.
    python_ver : bool
        Whether to show python version.
    pwd : bool
        Whether to show pwd.
    git : bool
        Whether to show git info.
    sys_info : bool, default True
        Whether to show platform info.

    Returns:
    -------
    dict
        Dictionary with module descriptions.
    """
    pyenv_info = {}

    python_version = sys.version_info[0]

    # get versions from __version__ string
    modules_versions = get_package_versions(packages)
    pyenv_info.update(modules_versions)

    # get versions from pip
    if type(pip_packages) == list and len(pip_packages) > 0 and pip_packages[0]:
        modules_versions_pip = get_pip_versions(pip_packages, python_version)
        pyenv_info.update(modules_versions_pip)

    if python_ver:
        # set python version
        try:
            pyenv_info["python"] = "{0}.{1}.{2}".format(*sys.version_info[0:3])
        except BaseException:
            pyenv_info["python"] = "unknown"

    if pwd:
        # set current path
        pyenv_info["pwd"] = os.path.dirname(os.path.realpath(__file__))

    if git:
        # set git revision of the code
        try:
            if os.name == "nt":
                command = "cmd /V /C \"cd {} && git log -n 1\"".format(pyenv_info["pwd"])
            else:
                command = ["cd {}; git log -n 1".format(pyenv_info["pwd"])]
            out_bytes = subprocess.check_output(command, shell=True)
            out_text = out_bytes.decode("utf-8")
        except BaseException:
            out_text = "unknown"
        pyenv_info["git"] = out_text.strip()

    if sys_info:
        pyenv_info["platform"] = platform.platform()

    return pyenv_info


def pretty_print_dict2str(d):
    """
    Pretty print of dictionary d to json-formated string.

    Parameters:
    ----------
    d : dict
        Dictionary with module descriptions.

    Returns:
    -------
    str
        Resulted string.
    """
    out_text = json.dumps(d, indent=4)
    return out_text


def get_env_stats(packages,
                  pip_packages,
                  python_ver=True,
                  pwd=True,
                  git=True,
                  sys_info=True):
    """
    Get environment statistics.

    Parameters:
    ----------
    packages : list of str
        list of package names to inspect only __version__.
    pip_packages : list of str
        List of package names to inspect by 'pip show'.
    python_ver : bool
        Whether to show python version.
    pwd : bool, default True
        Whether to show pwd.
    git : bool, default True
        Whether to show git info.
    sys_info : bool, default True
        Whether to show platform info.

    Returns:
    -------
    str
        Resulted string with information.
    """
    package_versions = get_pyenv_info(packages, pip_packages, python_ver, pwd, git, sys_info)
    return pretty_print_dict2str(package_versions)