import torch
import torch.nn as nn

from functools import partial


class Losses:
    
    def __init__(self,
                 loss:str = 'mse'):
        self._loss = self._get_loss(loss)
        self.loss_fn = self.masked_loss
        self.loss_name = loss
    
    def _get_loss(self, loss):
        return {
            'mse': nn.MSELoss(),
            'ccc': self.ccc,
            'ce': partial(self.cross_entropy_loss, nn.CrossEntropyLoss())
        }[loss]
    
    def masked_loss(self, predictions:torch.Tensor, labels:torch.Tensor, mask:torch.Tensor):
        
        tensor_device = predictions.get_device()
        tensor_dtype = predictions.dtype
        
        batch_preds = torch.tensor([0.0], dtype=tensor_dtype).to(tensor_device)
        batch_labs = torch.tensor([0.0], dtype=tensor_dtype).to(tensor_device)
        for i, m in enumerate(mask):
            batch_preds = torch.cat((batch_preds, predictions[i,:m].view(-1,)), 0)
            batch_labs = torch.cat((batch_labs, labels[i,:m].view(-1,)), 0)
        
        batch_preds = batch_preds[1:].view(-1,)
        batch_labs = batch_labs[1:].view(-1,)
        
        total_loss = self._loss(batch_preds.view(-1,), batch_labs.view(-1,))
        return total_loss
    
    def ccc(self, predictions, labels):
        
        def _get_moments(data_tensor):
            mean_t = torch.mean(data_tensor, 0)
            var_t = torch.var(data_tensor, 0, unbiased=False)
            return mean_t, var_t
        
        labels_mean, labels_var = _get_moments(labels)
        pred_mean, pred_var = _get_moments(predictions)
        
        mean_cent_prod = torch.mean((predictions - pred_mean) * (labels - labels_mean))
        batch_ccc = 1 - (2 * mean_cent_prod) / (pred_var + labels_var + torch.square(pred_mean - labels_mean))
        
        return torch.mean(batch_ccc)
    
    def cross_entropy_loss(self, instance_cross_entropy_loss, predictions, labels):
        predictions = predictions.unsqueeze(0)
        labels = labels.type(torch.long)
        return instance_cross_entropy_loss(predictions, labels)
