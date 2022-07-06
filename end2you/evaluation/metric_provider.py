import numpy as np
import torch

from sklearn.metrics import recall_score


class MetricProvider:
    
    def __init__(self,
                 metric:str = 'mse'):
        """ Helper class to use metric.
        
        Args:
            metric (str): Metric to use for evaluation.
        """
        
        self._metric = self._get_metric(metric) if metric else None
        self.eval_fn = self.masked_eval_fn
        self.metric_name = metric
    
    def _get_metric(self, metric:str):
        """ Factory method to return metric.
        
        Args:
            metric (str): Metric to use for evaluation.
        """
        
        return {
            'mse': self.MSE,
            'ccc': self.CCC,
            'uar': self.UAR
        }[metric.lower()]
    
    def masked_eval_fn(self, 
                       predictions:np.array, 
                       labels:np.array, 
                       masks:list,
                       take_last_frame:bool = True):
        """ Method to compute the masked metric evaluation.
        
        Args:
            predictions (np.array): Model predictions.
            labels (np.array): Data labels.
            masks (list): List of the frames to consider in each batch.
        """
        
        dtype = predictions[0].dtype
        
        batch_preds = []
        batch_labs = [] 
        
        for i, m in enumerate(masks):
            nframes = m - 1 if take_last_frame else range(m)
            batch_preds.append(predictions[i][nframes])
            batch_labs.append(labels[i][nframes])
        
        return self._metric(batch_preds, batch_labs)
    
    def CCC(self, predictions:list, labels:list):
        """ Concordance Correlation Coefficient (CCC) metric.
        
        Args:
            predictions (list): Model predictions.
            labels (list): Data labels.
        """
        
        predictions = np.concatenate(predictions).reshape(-1,)
        labels = np.concatenate(labels).reshape(-1,)
        
        mean_cent_prod = ((predictions - predictions.mean()) * (labels - labels.mean())).mean()
        return (2 * mean_cent_prod) / (predictions.var() + labels.var() + (predictions.mean() - labels.mean()) ** 2)
    
    def UAR(self, predictions:list, labels:list):
        """ Unweighted Average Recall (UAR) metric.
        
        Args:
            predictions (list): Model predictions.
            labels (list): Data labels.
        """
        predictions = np.stack(predictions)
        labels = np.stack(labels)
        
#         predictions = np.stack([x[...,-1,:] for x in predictions])
#         labels = np.stack([x[...,-1,:] for x in labels])
        
        predictions = np.argmax(predictions, axis = 1)
        
        predictions = predictions.astype(np.int32).reshape(-1,).tolist()
        labels = labels.astype(np.int32).reshape(-1,).tolist()
        
        return recall_score(labels, predictions, average="macro")
    
    def MSE(self, predictions:list, labels:list):
        """ Mean Squared Error (MSE) metric.
        
        Args:
            predictions (list): Model predictions.
            labels (list): Data labels.
        """
        
        predictions = np.concatenate(predictions).reshape(-1,)
        labels = np.concatenate(labels).reshape(-1,)
        
        mse_score = np.mean((predictions - labels)**2)
        
        return mse_score.astype(np.float64)
    