import torch
import torch.nn.functional as F

def guess_label(self, classifier, y, T, **kwargs):
    del kwargs
    logits_y = [classifier(yi, training=True) for yi in y]
    logits_y = torch.cat(logits_y, 0)
    
    # Compute predicted probability distribution py.
    p_model_y = F.softmax(logits_y, dim=1)
    p_model_y = torch.mean(p_model_y.view(len(y), -1, self.nclass), dim=0)
    
    # Compute the target distribution.
    p_target = torch.pow(p_model_y, 1. / T)
    p_target /= torch.sum(p_target, dim=1, keepdim=True)
    
    return {'p_target': p_target, 'p_model': p_model_y}