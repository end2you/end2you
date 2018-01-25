import tensorflow as tf

from .model import Model


class RNNModel(Model):
    
    def __init__(self,
                 num_layers:int = 2,
                 hidden_units:int = 128,
                 bidirectional:bool = False,
                 cell_type:str = 'GRU'):
        
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.bidirectional = bidirectional
        
        if ('gru' and 'lstm') not in cell_type.lower():
            raise ValueError('Cell type should be one of GRU or LSTM. \
                             [{}] found'.format(cell_type))
        
        self.cell_type = cell_type.lower()
        
    def create_model(self, inputs):
        
        with tf.variable_scope("recurrent"):
            batch_size, seq_length, num_features = inputs.get_shape().as_list()
            
            if 'gru' in self.cell_type:
                rnn_cell = tf.contrib.rnn.GRUCell(self.hidden_units)
            else:
                rnn_cell = tf.contrib.rnn.LSTMCell(self.hidden_units,
                                                  use_peepholes=True,
                                                  cell_clip=100,
                                                  state_is_tuple=True)
        
            stacked_cells = tf.contrib.rnn.MultiRNNCell([rnn_cell], #  * self.num_layers 
                                                       state_is_tuple=True)
            
            outputs, _ = tf.nn.dynamic_rnn(stacked_cells, inputs, dtype=tf.float32)

        if seq_length == None:
            seq_length = -1
        
        net = tf.reshape(outputs, (batch_size, seq_length, self.hidden_units))
        
        return net
