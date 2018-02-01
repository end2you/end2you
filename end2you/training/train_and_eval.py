import tensorflow as tf
import numpy as np
import time
import os
import shutil
import glob
import csv
import sys
sys.path.append("..")

from ..evaluation import *
from pathlib import Path
from .training import Train


slim = tf.contrib.slim

class TrainEval(Train):
    
    def __init__(self, *args, **kwargs):
        self.tfrecords_eval_folder = kwargs['tfrecords_eval_folder']
        kwargs.pop('tfrecords_eval_folder')
        
        super().__init__(*args, **kwargs)
        
        self.log_dir = Path(self.train_dir) / 'log'
        self.train_dir = str(Path(self.train_dir) / 'train')
        
        self.save_top_k = 5 #kwargs['save_top_k']
        self.save_dir = Path(self.train_dir) / 'top_saved_models'
        
        self.metric = kwargs['loss'].lower()
        if 'sce' in self.metric or 'ce' in self.metric:
            self.metric = 'uar'
        
        old_perf_models = str(self.save_dir / "models_performance.txt")
        if Path(old_perf_models).exists():
            r = list(csv.reader(open(old_perf_models, "r"), delimiter=':'))
            self.best_perfs = {x[0]:float(x[1]) for x in r}
        else:
            self.best_perfs = {str(x):float('-inf') for x in np.arange(self.save_top_k)}
            os.system('mkdir -p {}'.format(self.save_dir))
        
        
    def _restore_variables(self, sess, saver):
        model_path = tf.train.latest_checkpoint(self.train_dir)
        if model_path != None:
            saver.restore(sess, model_path)
            step = int(model_path.split('-')[1])
            print('Variables restored from [{}]'.format(model_path))
            return step
        return 0
    
    def _save_best_model(self, best_perfs, model_perf):
        model_path = tf.train.latest_checkpoint(self.train_dir)
        if model_path == None:
            raise ValueError('No model to test performance and save.')
        model_path = Path(model_path)
        
        vals = np.array(list(best_perfs.values()))
        min_val = np.min(vals)
        if min_val < model_perf:
            min_idx = np.argmin(vals)
            
            # delete prev model from folder and dict
            prev_model = list(best_perfs.keys())[min_idx]
            if os.path.exists(prev_model + '.meta'):
                os.system('rm {}'.format(prev_model + '*'))
            best_perfs.pop(prev_model)

            # insert new model to folder and dict
            new_model_name = model_path.name #os.path.basename(model_path)
            current_model = Path(self.save_dir) / new_model_name
            best_perfs[str(current_model)] = model_perf
            
            for filename in glob.glob(str(model_path) + '*'):
                shutil.copy(str(filename), str(self.save_dir))
        
        return best_perfs
    
    def _del_first_model(self, save_model_path):
        list_of_files = glob.glob(str(save_model_path) + '*.meta')
        if len(list_of_files) > self.save_top_k:
            latest_file = Path(min(list_of_files, key=os.path.getctime))
            parts = latest_file.name.split('-')
            model_file = parts[0] + '-' + parts[1][0]
            rm_files = latest_file.parent / model_file
            os.system('rm {}'.format(str(rm_files) + '*'))
    
    def _eval_and_sum(self, sess, eval_pred, eval_labs, eval_batches, eval_summary_writer, step):
        total_eval = EvalOnce.eval_once(
            sess, eval_pred, eval_labs, eval_batches, self.num_outputs, self.metric)
                
        tf_eval_sum = tf.summary.scalar('eval/total eval', tf.convert_to_tensor(total_eval))
        eval_sum = sess.run(tf_eval_sum)

        eval_summary_writer.add_summary(eval_sum, global_step=step)
        
        print('\n End of evaluation. Result: UAR {}'.format(total_eval))
        
        return total_eval
    
    def start_training(self):
        
        best_perfs = self.best_perfs
        
        g = tf.get_default_graph()
        frames, labels, sids = self.data_provider.get_batch()
        
        if self.data_provider.seq_length != 0:
            frames_shape = frames.get_shape().as_list()
            batch = self.data_provider.seq_length * self.data_provider.batch_size
            frames = tf.reshape(frames, [batch, *frames_shape[2:]] )
        
        predictions = self.predictions(frames)
        loss = self.set_train_loss(predictions, labels)
        total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('losses/total loss', total_loss)
        
        optimizer = tf.train.AdamOptimizer(self.initial_learning_rate)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            
            train_op = slim.learning.create_train_op(total_loss,
                                                     optimizer,
                                                     summarize_gradients=True)
            
            seq_length = 1 if self.data_provider.seq_length == 0 \
                else self.data_provider.seq_length
            num_batches = int(np.ceil(
                    self.data_provider.num_examples / (self.data_provider.batch_size * seq_length)))
            
            summary_writer = tf.summary.FileWriter(str(self.train_dir), graph=g)
            eval_summary_writer = tf.summary.FileWriter(str(self.log_dir), graph=g)
            merged_summaries = tf.summary.merge_all()
            
            get_eval_once = EvalOnce.get_eval_tensors(sess, self.predictions, self.data_provider, 
                                                  self.tfrecords_eval_folder)
            eval_pred, eval_labs, eval_batches = get_eval_once
            
            saver = tf.train.Saver()
            step = self._restore_variables(sess, saver)
            
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            save_model_path = Path(self.train_dir) / 'model.ckpt'
            
            # Start training of the model
            for epoch in range(self.num_epochs):
                print('\n Start Training for epoch {}\n'.format(epoch + 1))
                for batch in range(num_batches):
                    start_time = time.time()
                    
                    _, step_loss, step_summary = sess.run([train_op, total_loss, merged_summaries])

                    time_step = time.time() - start_time
                    
                    print("Epoch {}/{}: Batch {}/{}: loss = {:.4f} ({:.2f} sec/step)".format(
                        epoch + 1, self.num_epochs, batch + 1, num_batches, step_loss, time_step))

                    summary_writer.add_summary(step_summary, global_step=step)
                    step += 1
                
                print('\n End of epoch {}'.format(epoch+1))
                
                # Start evaluation in the end of each epoch
                total_eval = self._eval_and_sum(
                    sess, eval_pred, eval_labs, eval_batches, eval_summary_writer, step)
                
                # Save model
                saver.save(sess, str(save_model_path), global_step=step)
                
                # Delete first saved model from train_dir
                self._del_first_model(save_model_path)
                
                best_perfs = self._save_best_model(best_perfs, total_eval)
            
            # Save performance of the top k models in text file
            w = csv.writer(open(str(self.save_dir / "models_performance.txt"), "w"), delimiter=':')
            for key, val in best_perfs.items():
                w.writerow([key, val])
            
            coord.request_stop()

            coord.join(threads)
            summary_writer.close()
            eval_summary_writer.close()
