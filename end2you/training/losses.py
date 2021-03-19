import torch
import torch.nn as nn

from functools import partial


class Losses:
    
    def __init__(self,
                 loss:str = 'mse'):
        """ Initialize loss object class.
        
        Args:
          loss (str): Loss function to use
        """
        
        self._loss = self._get_loss(loss)
        self.loss_fn = self.masked_loss
        self.loss_name = loss
    
    def _get_loss(self, loss:str):
        """ Factory method to provide the loss function.
        
        Args:
          loss (str): Name of loss function to use.
        """
        
        return {
            'mse': nn.MSELoss(),
            'ccc': self.ccc,
            'ce': partial(self.cross_entropy_loss, nn.CrossEntropyLoss())
        }[loss]
    
    def masked_loss(self, 
                    predictions:torch.Tensor, 
                    labels:torch.Tensor, 
                    mask:torch.Tensor):
        """ Method that computes a masked loss, meaning that it does not
            include the all samples in the predictions/labels tensor. 
        
        Args:
          predictions (torch.Tensor): Predictions of the model.
          labels (torch.Tensor): Labels of the data.
          mask (torch.Tensor): Tensor with the samples to consider in the batch.
        """
        
        tensor_device = predictions.get_device()
        if tensor_device == -1:
            tensor_device = torch.device('cpu')
        tensor_dtype = predictions.dtype
        
        batch_preds = []
        batch_labs = []
        for i, m in enumerate(mask):
            batch_preds.extend(predictions[i,:m]) 
            batch_labs.extend(labels[i,:m]) 
        
        batch_preds = torch.stack(batch_preds)
        batch_labs = torch.stack(batch_labs)
        
        return self._loss(batch_preds, batch_labs)
    
    def ccc(self, 
            predictions:torch.Tensor, 
            labels:torch.Tensor):
        """ Concordance Correlation Coefficient (CCC) loss. 
        
        Args:
          predictions (torch.Tensor): Predictions of the model.
          labels (torch.Tensor): Labels of the data.
        """
        
        predictions = predictions.view(-1,)
        labels = labels.view(-1,)
        
        def _get_moments(data_tensor):
            mean_t = torch.mean(data_tensor)#, 0)
            var_t = torch.var(data_tensor)#, 0, unbiased=False)
            return mean_t, var_t
        
        labels_mean, labels_var = _get_moments(labels)
        pred_mean, pred_var = _get_moments(predictions)
        
        mean_cent_prod = torch.mean((predictions - pred_mean) * (labels - labels_mean))
        batch_ccc = 1 - (2 * mean_cent_prod) / (pred_var + labels_var + torch.square(pred_mean - labels_mean))
        
        return batch_ccc
    
    def cross_entropy_loss(self, 
                           instance_cross_entropy_loss,
                           predictions:torch.Tensor, 
                           labels:torch.Tensor):
        """ Cross Entropy loss. 
        
        Args:
          instance_cross_entropy_loss: Instance of cross entropy loss, i.e., nn.CrossEntropyLoss())
          predictions (torch.Tensor): Predictions of the model.
          labels (torch.Tensor): Labels of the data.
        """
        
        labels = labels.view(-1).type(torch.long)
        return instance_cross_entropy_loss(predictions, labels)
