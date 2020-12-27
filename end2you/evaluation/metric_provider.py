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
    
    def masked_eval_fn(self, predictions:np.array, labels:np.array, masks:list):
        
        dtype = predictions[0].dtype
        
        batch_preds = []
        batch_labs = [] 
        
        for i, m in enumerate(masks):
            batch_preds.append(predictions[i][:m])
            batch_labs.append(labels[i][:m])
        
        return self._metric(batch_preds, batch_labs)
    
    def CCC(self, predictions:np.array, labels:np.array):
        predictions = np.stack(predictions).reshape(-1,)
        labels = np.stack(labels).reshape(-1,)
        
        mean_cent_prod = ((predictions - predictions.mean()) * (labels - labels.mean())).mean()
        return (2 * mean_cent_prod) / (predictions.var() + labels.var() + (predictions.mean() - labels.mean()) ** 2)
    
    def UAR(self, predictions:np.array, labels:np.array):
        predictions = np.stack([x[...,-1,:] for x in predictions])
        labels = np.stack([x[...,-1,:] for x in labels])
        
        predictions = np.argmax(predictions, axis = 1)
        
        predictions = predictions.astype(np.int32).reshape(-1,).tolist()
        labels = labels.astype(np.int32).reshape(-1,).tolist()
        
        return recall_score(labels, predictions, average="macro")
    
    def MSE(self, predictions:np.array, labels:np.array):
        return np.mean((predictions - labels)**2)
    