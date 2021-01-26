import collections

import tensorflow as tf
import tensorflow.keras as keras


LOG_DICT = None
def start_logging():
    global LOG_DICT
    LOG_DICT = collections.defaultdict(keras.metrics.Mean)
    
def get_logs():
    res = {k:LOG_DICT[k].result() for k in LOG_DICT}
    for m in LOG_DICT.values():
        m.reset_states()
    return res

class SPP_layer(keras.layers.Layer):
    def __init__(self, name):
        super(SPP_layer, self).__init__(name = name)
    def call(self,feature_map):
        if LOG_DICT is not None:
            avg_squared_channel_mean = tf.reduce_mean(feature_map, axis=(0,1,2))
            avg_squared_channel_mean = avg_squared_channel_mean*avg_squared_channel_mean       
            avg_squared_channel_mean = tf.reduce_mean(avg_squared_channel_mean)
            
            avg_channel_std = tf.math.reduce_std(feature_map, axis=(0,1,2))
            avg_channel_std = tf.reduce_mean(avg_channel_std)
            
            LOG_DICT[f"{self.name}_mean"].update_state(avg_squared_channel_mean)
            LOG_DICT[f"{self.name}_std"].update_state(avg_channel_std)
            
        return feature_map 