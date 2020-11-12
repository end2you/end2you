import numpy as np
import torch

from sklearn.metrics import recall_score


class MetricProvider:
    
    def __init__(self,
                 metric:str = 'mse'):
        ''' Helper class to use metric.
        Args:
            metric (str): Metric to use for evaluation.
        '''
        self._metric = self._get_metric(metric)
        self.eval_fn = self.masked_eval_fn
        self.metric_name = metric
    
    def _get_metric(self, metric:str):
        return {
            'mse': self.MSE,
            'ccc': self.CCC,
            'uar': self.UAR
        }[metric.lower()]
    
    def masked_eval_fn(self, predictions:torch.Tensor, labels:torch.Tensor, mask:torch.Tensor):
        
        predictions = predictions.data.cpu().numpy()
        labels = labels.data.cpu().numpy()
        
        num_samples = len(mask)
        score = 0
        for i in range(num_samples):
            m = mask[i]
            score += self._metric(predictions[i,:m].reshape(-1,), labels[i,:m].reshape(-1,))
        
        return score / num_samples
    
    def CCC(self, predictions:np.array, labels:np.array):
        mean_cent_prod = ((predictions - predictions.mean()) * (labels - labels.mean())).mean()
        return (2 * mean_cent_prod) / (predictions.var() + labels.var() + (predictions.mean() - labels.mean()) ** 2)
    
    def UAR(self, predictions:np.array, labels:np.array):
        predictions = np.argmax(predictions, axis = 1)
        
        return recall_score(labels, predictions, average="macro")
    
    def MSE(self, predictions:np.array, labels:np.array):
        return np.mean((predictions - labels)**2)
    