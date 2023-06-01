"""
    Main utils file for the project

    code from https://github.com/osmr/imgclsmob/blob/master/pytorch/metrics/cls_metrics.py, https://github.com/osmr/imgclsmob/blob/master/pytorch/utils.py#L282

"""

__all__ = ['EvalMetric','Top1Error', 'TopKError']


from collections import OrderedDict
import numpy as np

import torch


def get_metric(metric_name, metric_extra):
    """
    Get metric.

    Parameters:
    ----------
    metric_name : str
        Metric name.
    metric_extra : dict
        Metric extra params.
    
    Returns:
    -------
    callable Metric.
    """
    if metric_name == "Top1Error":
        return Top1Error(**metric_extra)
    elif metric_name == "TopKError":
        return TopKError(**metric_extra)
    else:
        raise Exception("Wrong metric name: {}".format(metric_name))
    
def get_composite_metric(metric_names, metric_extra_kwargs):
    """
    Get composite metric by list of metric names.

    Parameters:
    ----------
    metric_names : list of str
        Metric name list.
    metric_extra_kwargs : list of dict
        Metric extra parameters list.

    Returns:
    -------
    CompositeEvalMetric
        Metric object instance.
    """
    if len(metric_names) == 1:
        metric = get_metric(metric_names[0], metric_extra_kwargs[0])
    else:
        metric = CompositeEvalMetric()
        for name, extra_kwargs in zip(metric_names, metric_extra_kwargs):
            metric.add(get_metric(name, extra_kwargs))
    return metric


class EvalMetric(object):
    """
    Base class for all evaluation metrics.

    Parameters:
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self,
                 name,
                 output_names=None,
                 label_names=None,
                 **kwargs):
        super(EvalMetric, self).__init__()
        self.name = str(name)
        self.output_names = output_names
        self.label_names = label_names
        self._has_global_stats = kwargs.pop("has_global_stats", False)
        self._kwargs = kwargs
        self.reset()

    def __str__(self):
        return "EvalMetric: {}".format(dict(self.get_name_value()))

    def get_config(self):
        """
        Save configurations of metric. Can be recreated from configs with metric.create(**config).
        """
        config = self._kwargs.copy()
        config.update({
            "metric": self.__class__.__name__,
            "name": self.name,
            "output_names": self.output_names,
            "label_names": self.label_names})
        return config

    def update_dict(self, label, pred):
        """
        Update the internal evaluation with named label and pred.

        Parameters:
        ----------
        labels : OrderedDict of str -> torch.Tensor
            name to array mapping for labels.
        preds : OrderedDict of str -> torch.Tensor
            name to array mapping of predicted outputs.
        """
        if self.output_names is not None:
            pred = [pred[name] for name in self.output_names]
        else:
            pred = list(pred.values())

        if self.label_names is not None:
            label = [label[name] for name in self.label_names]
        else:
            label = list(label.values())

        self.update(label, pred)

    def update(self, labels, preds):
        """
        Updates the internal evaluation result.

        Parameters:
        ----------
        labels : torch.Tensor
            The labels of the data.
        preds : torch.Tensor
            Predicted values.
        """
        raise NotImplementedError()

    def reset(self):
        """
        Resets the internal evaluation result to initial state.
        """
        self.num_inst = 0
        self.sum_metric = 0.0
        self.global_num_inst = 0
        self.global_sum_metric = 0.0

    def reset_local(self):
        """
        Resets the local portion of the internal evaluation results to initial state.
        """
        self.num_inst = 0
        self.sum_metric = 0.0

    def get(self):
        """
        Gets the current evaluation result.

        Returns:
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num_inst == 0:
            return self.name, float("nan")
        else:
            return self.name, self.sum_metric / self.num_inst

    def get_global(self):
        """
        Gets the current global evaluation result.

        Returns:
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self._has_global_stats:
            if self.global_num_inst == 0:
                return self.name, float("nan")
            else:
                return self.name, self.global_sum_metric / self.global_num_inst
        else:
            return self.get()

    def get_name_value(self):
        """
        Returns zipped name and value pairs.

        Returns:
        -------
        list of tuples
            A (name, value) tuple list.
        """
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))

    def get_global_name_value(self):
        """
        Returns zipped name and value pairs for global results.

        Returns:
        -------
        list of tuples
            A (name, value) tuple list.
        """
        if self._has_global_stats:
            name, value = self.get_global()
            if not isinstance(name, list):
                name = [name]
            if not isinstance(value, list):
                value = [value]
            return list(zip(name, value))
        else:
            return self.get_name_value()


class CompositeEvalMetric(EvalMetric):
    """
    Manages multiple evaluation metrics.

    Parameters:
    ----------
    name : str, default 'composite'
        Name of this metric instance for display.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """

    def __init__(self,
                 name="composite",
                 output_names=None,
                 label_names=None):
        super(CompositeEvalMetric, self).__init__(
            name,
            output_names=output_names,
            label_names=label_names,
            has_global_stats=True)
        self.metrics = []

    def add(self, metric):
        """
        Adds a child metric.

        Parameters:
        ----------
        metric
            A metric instance.
        """
        self.metrics.append(metric)

    def update_dict(self, labels, preds):
        if self.label_names is not None:
            labels = OrderedDict([i for i in labels.items()
                                  if i[0] in self.label_names])
        if self.output_names is not None:
            preds = OrderedDict([i for i in preds.items()
                                 if i[0] in self.output_names])

        for metric in self.metrics:
            metric.update_dict(labels, preds)

    def update(self, labels, preds):
        """
        Updates the internal evaluation result.

        Parameters:
        ----------
        labels : torch.Tensor
            The labels of the data.

        preds : torch.Tensor
            Predicted values.
        """
        for metric in self.metrics:
            metric.update(labels, preds)

    def reset(self):
        """
        Resets the internal evaluation result to initial state.
        """
        try:
            for metric in self.metrics:
                metric.reset()
        except AttributeError:
            pass

    def reset_local(self):
        """
        Resets the local portion of the internal evaluation results to initial state.
        """
        try:
            for metric in self.metrics:
                metric.reset_local()
        except AttributeError:
            pass

    def get(self):
        """
        Returns the current evaluation result.

        Returns:
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        names = []
        values = []
        for metric in self.metrics:
            name, value = metric.get()
            name = [name]
            value = [value]
            names.extend(name)
            values.extend(value)
        return names, values

    def get_global(self):
        """
        Returns the current evaluation result.

        Returns:
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        names = []
        values = []
        for metric in self.metrics:
            name, value = metric.get_global()
            name = [name]
            value = [value]
            names.extend(name)
            values.extend(value)
        return names, values

    def get_config(self):
        config = super(CompositeEvalMetric, self).get_config()
        config.update({"metrics": [i.get_config() for i in self.metrics]})
        return config


class Accuracy(EvalMetric):
    """
    Computes accuracy classification score.

    Parameters:
    ----------
    axis : int, default 1
        The axis that represents classes
    name : str, default 'accuracy'
        Name of this metric instance for display.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self,
                 axis=1,
                 name="accuracy",
                 output_names=None,
                 label_names=None):
        super(Accuracy, self).__init__(
            name,
            axis=axis,
            output_names=output_names,
            label_names=label_names,
            has_global_stats=True)
        self.axis = axis

    def update(self, labels, preds):
        """
        Updates the internal evaluation result.

        Parameters:
        ----------
        labels : torch.Tensor
            The labels of the data with class indices as values, one per sample.
        preds : torch.Tensor
            Prediction values for samples. Each prediction value can either be the class index,
            or a vector of likelihoods for all classes.
        """
        assert (len(labels) == len(preds))
        with torch.no_grad():
            if preds.shape != labels.shape:
                pred_label = torch.argmax(preds, dim=self.axis)
            else:
                pred_label = preds
            pred_label = pred_label.cpu().numpy().astype(np.int32)
            label = labels.cpu().numpy().astype(np.int32)

            label = label.flat
            pred_label = pred_label.flat

            num_correct = (pred_label == label).sum()
            self.sum_metric += num_correct
            self.global_sum_metric += num_correct
            self.num_inst += len(pred_label)
            self.global_num_inst += len(pred_label)


class TopKAccuracy(EvalMetric):
    """
    Computes top k predictions accuracy.

    Parameters:
    ----------
    top_k : int, default 1
        Whether targets are in top k predictions.
    name : str, default 'top_k_accuracy'
        Name of this metric instance for display.
    torch_like : bool, default True
        Whether to use pytorch-like algorithm.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self,
                 top_k=1,
                 name="top_k_accuracy",
                 torch_like=True,
                 output_names=None,
                 label_names=None):
        super(TopKAccuracy, self).__init__(
            name,
            top_k=top_k,
            output_names=output_names,
            label_names=label_names,
            has_global_stats=True)
        self.top_k = top_k
        assert (self.top_k > 1), "Please use Accuracy if top_k is no more than 1"
        self.name += "_{:d}".format(self.top_k)
        self.torch_like = torch_like

    def update(self, labels, preds):
        """
        Updates the internal evaluation result.

        Parameters:
        ----------
        labels : torch.Tensor
            The labels of the data.
        preds : torch.Tensor
            Predicted values.
        """
        assert (len(labels) == len(preds))
        with torch.no_grad():
            if self.torch_like:
                _, pred = preds.topk(k=self.top_k, dim=1, largest=True, sorted=True)
                pred = pred.t()
                correct = pred.eq(labels.view(1, -1).expand_as(pred))
                # num_correct = correct.view(-1).float().sum(dim=0, keepdim=True).item()
                num_correct = correct.flatten().float().sum(dim=0, keepdim=True).item()
                num_samples = labels.size(0)
                assert (num_correct <= num_samples)
                self.sum_metric += num_correct
                self.global_sum_metric += num_correct
                self.num_inst += num_samples
                self.global_num_inst += num_samples
            else:
                assert(len(preds.shape) <= 2), "Predictions should be no more than 2 dims"
                pred_label = preds.cpu().numpy().astype(np.int32)
                pred_label = np.argpartition(pred_label, -self.top_k)
                label = labels.cpu().numpy().astype(np.int32)
                assert (len(label) == len(pred_label))
                num_samples = pred_label.shape[0]
                num_dims = len(pred_label.shape)
                if num_dims == 1:
                    num_correct = (pred_label.flat == label.flat).sum()
                    self.sum_metric += num_correct
                    self.global_sum_metric += num_correct
                elif num_dims == 2:
                    num_classes = pred_label.shape[1]
                    top_k = min(num_classes, self.top_k)
                    for j in range(top_k):
                        num_correct = (pred_label[:, num_classes - 1 - j].flat == label.flat).sum()
                        self.sum_metric += num_correct
                        self.global_sum_metric += num_correct
                self.num_inst += num_samples
                self.global_num_inst += num_samples


class Top1Error(Accuracy):
    """
    Computes top-1 error (inverted accuracy classification score).

    Parameters:
    ----------
    axis : int, default 1
        The axis that represents classes.
    name : str, default 'top_1_error'
        Name of this metric instance for display.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self,
                 axis=1,
                 name="top_1_error",
                 output_names=None,
                 label_names=None):
        super(Top1Error, self).__init__(
            axis=axis,
            name=name,
            output_names=output_names,
            label_names=label_names)

    def get(self):
        """
        Gets the current evaluation result.

        Returns:
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num_inst == 0:
            return self.name, float("nan")
        else:
            return self.name, 1.0 - self.sum_metric / self.num_inst


class TopKError(TopKAccuracy):
    """
    Computes top-k error (inverted top k predictions accuracy).

    Parameters:
    ----------
    top_k : int
        Whether targets are out of top k predictions, default 1
    name : str, default 'top_k_error'
        Name of this metric instance for display.
    torch_like : bool, default True
        Whether to use pytorch-like algorithm.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self,
                 top_k=1,
                 name="top_k_error",
                 torch_like=True,
                 output_names=None,
                 label_names=None):
        name_ = name
        super(TopKError, self).__init__(
            top_k=top_k,
            name=name,
            torch_like=torch_like,
            output_names=output_names,
            label_names=label_names)
        self.name = name_.replace("_k_", "_{}_".format(top_k))

    def get(self):
        """
        Gets the current evaluation result.

        Returns:
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num_inst == 0:
            return self.name, float("nan")
        else:
            return self.name, 1.0 - self.sum_metric / self.num_inst