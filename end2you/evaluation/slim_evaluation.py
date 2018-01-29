import tensorflow as tf

from .metrics import tf_concordance_cc2
from .evaluation import Eval


class SlimEvaluation(Eval):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_mse(self, predictions, labels):
        
        metric = {}
        for i, name in enumerate(self.name_pred):
            key = 'eval/{}'.format(name)
            metric[key] = slim.metrics.streaming_mean_squared_error(predictions[:,:,i], 
                                                                    labels[:,:,i])
            
        summary_ops = []
        mse_total = 0
        
        # Computing MSE and Concordance values, and adding them to summary
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(metric)
        for i, name in enumerate(self.name_pred):
            key = 'eval/{}'.format(name)
            op = self._create_summary(key, names_to_values[key])
            summary_ops.append(op)
        
            mse_total += names_to_values[key]
        
        mse_total = mse_total / self.num_outputs
        
        op = self._create_summary('eval/mse_total', mse_total)
        summary_ops.append(op)
        
        return names_to_values, names_to_updates, summary_ops
    
    def get_ccc(self, predictions, labels):
        
        summary_ops = []
        names_to_updates = {}
        conc_total = 0
        for i, name in enumerate(self.name_pred):
            with tf.name_scope(name) as scope:
                concordance_cc2, values, updates = tf_concordance_cc2(
                            tf.reshape(predictions[:,:,i], [-1]),
                            tf.reshape(labels[:,:,i], [-1]))
            
            for n, v in updates.items():
                names_to_updates[n + '/' + name] = v
          
            op = self._create_summary('eval/concordance_' + name, concordance_cc2)
            summary_ops.append(op)
          
            conc_total += concordance_cc2

        conc_total = conc_total / self.num_outputs
        
        op = self._create_summary('eval/concordance_total', conc_total)
        summary_ops.append(op)
        
        return {}, names_to_updates, summary_ops
    
    def get_uar(self, predictions, labels):
        pred_argmax = tf.argmax(predictions, 1)
        lab_argmax = tf.argmax(labels, 1)

        metrics = {
          "eval/accuracy": slim.metrics.streaming_accuracy(pred_argmax, lab_argmax, name='accuracy')
        }

        for i in range(self.num_outputs):
            name ='eval/recall_{}'.format(i)
            recall = slim.metrics.streaming_recall(
                  tf.to_int64(tf.equal(pred_argmax, i)),
                  tf.to_int64(tf.equal(lab_argmax, i)), name=name)
            metrics[name] = recall

        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(metrics)

        summary_ops = []
        metrics = dict()
        for name, value in names_to_values.items():
            op = tf.summary.scalar(name, value)
            op = tf.Print(op, [value], name)
            summary_ops.append(op)
            metrics[name] = value

        # Computing the unweighted average recall and add it into the summaries.
        uar = sum([metrics['eval/recall_{}'.format(i)] \
                   for i in range(self.num_outputs)]) / self.num_outputs
        
        op = tf.summary.scalar('eval/uar', uar)
        op = tf.Print(op, [uar], 'eval/uar')
        summary_ops.append(op)
        
        return {}, names_to_updates, summary_ops
    
    @staticmethod
    def get_metric(metric):
        return {
            'ccc':self.get_ccc,
            'mse':self.get_mse,
            'uar':self.get_uar
        }[metric]
    
    def start_evaluation(self):
        
        frames, labels, sids = self.data_provider.get_batch()
        
        predictions = self.predictions(frames)

        names_to_values = {}
        names_to_updates = {}
        summary_ops = []
        for m in self.metric:
            values, updates, s_ops = self.get_metric(m)(predictions, labels)
            names_to_values.update(values)
            names_to_updates.update(updates)
            summary_ops.extend(s_ops)
        
        seq_length = 1 if self.data_provider.seq_length == 0 \
                       else self.data_provider.seq_length
            
        num_batches = int(self.num_examples / (1 * seq_length))
        logging.set_verbosity(1)

        # Setup the global step.
        eval_interval_secs = self.eval_interval_secs # How often to run the evaluation.
        slim.evaluation.evaluation_loop(
            '',
            self.train_dir,
            self.log_dir,
            num_evals=num_batches,
            eval_op=list(names_to_updates.values()),
            summary_op=tf.summary.merge(summary_ops),
            eval_interval_secs=eval_interval_secs)