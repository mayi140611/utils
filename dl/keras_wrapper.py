#!/usr/bin/python
# encoding: utf-8
# version: Keras==2.2.2

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

from keras.layers import Input
from keras.models import Model



class keras_wrapper(object):
    def __init__(self):
        pass
    '''
    建立sequential模型
    '''
    @classmethod
    def build_sequential_nn(self):
        return Sequential()
    @classmethod
    def add_dense_layer(self,model,units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_dim=None):
        '''
        添加全连接层
        
        units: 全连接层神经元个数。Positive integer, dimensionality of the output space.
        input_dim: 输入的神经元个数。如果是第一层需要制定，否则不需要！
            如：model.add(Dense(units=64, input_dim=100))
        activation: Activation function to use (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
            可选项：'relu','tanh','softmax'
        '''
        return model.add(Dense(units=units,activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, input_dim))
    @classmethod
    def add_masking_layer(self,model,mask_value=0.0):
        '''
        使用给定的值对输入的序列信号进行“屏蔽”，用以定位需要跳过的时间步
        考虑输入数据x是一个形如(samples,timesteps,features)的张量，现将其送入LSTM层。因为你缺少时间步为3和5的信号，所以你希望将其掩盖。这时候应该：

        赋值x[:,3,:] = 0.，x[:,5,:] = 0.
        在LSTM层之前插入mask_value=0.的Masking层
        '''
        return model.add(Dense(units=units,activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, input_dim))
    '''
    建立Functional模型
    '''
    @classmethod
    def add_functional_input_layer(self,shape=None, batch_shape=None, name=None, dtype=None, sparse=False, tensor=None):
        '''
        添加输入层
        '''
        return Input(shape, batch_shape, name, dtype, sparse, tensor)
    
    @classmethod
    def add_functional_dense_layer(self,prelayer,units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_dim=None):
        '''
        添加全连接层
        
        prelayer: 前一层
        units: 全连接层神经元个数。Positive integer, dimensionality of the output space.
        input_dim: 输入的神经元个数。如果是第一层需要制定，否则不需要！
            如：model.add(Dense(units=64, input_dim=100))
        activation: Activation function to use (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
            可选项：'relu','tanh','softmax'
        '''
        return Dense(units=units,activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, input_dim)(prelayer)
    @classmethod
    def build_functional_nn(self,inputs, outputs):
        return Model(inputs=inputs, outputs=outputs)
    
    
    
    @classmethod
    def compile(self,model, optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None):
        '''
        Configures the model for training.
        
        optimizer: String (name of optimizer) or optimizer instance.
            可选值:'sgd'(随机梯度),'rmsprop','adam'
        loss: String (name of objective function) or objective function.
            可选值:'categorical_crossentropy'
            If the model has multiple outputs, you can use a different loss
            on each output by passing a dictionary or a list of losses.
            The loss value that will be minimized by the model
            will then be the sum of all individual losses.
        metrics: List of metrics to be evaluated by the model
            during training and testing.
            Typically you will use `metrics=['accuracy']`.
            To specify different metrics for different outputs of a
            multi-output model, you could also pass a dictionary,
            such as `metrics={'output_a': 'accuracy'}`.
        '''
        return model.compile(optimizer, loss, metrics, loss_weights, sample_weight_mode, weighted_metrics, target_tensors)
    @classmethod
    def fit(self,model,x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None):
        '''
        Trains the model for a given number of epochs (iterations on a dataset).
        
        x: Numpy array of training data (if the model has a single input),
            or list of Numpy arrays (if the model has multiple inputs).
            If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
            `x` can be `None` (default) if feeding from
            framework-native tensors (e.g. TensorFlow data tensors).
        y: Numpy array of target (label) data
            (if the model has a single output),
            or list of Numpy arrays (if the model has multiple outputs).
            If output layers in the model are named, you can also pass a
            dictionary mapping output names to Numpy arrays.
            `y` can be `None` (default) if feeding from
            framework-native tensors (e.g. TensorFlow data tensors).
        batch_size: Integer or `None`.
            Number of samples per gradient update.
            If unspecified, `batch_size` will default to 32.
        epochs: Integer. Number of epochs to train the model.
            An epoch is an iteration over the entire `x` and `y`
            data provided.
            Note that in conjunction with `initial_epoch`,
            `epochs` is to be understood as "final epoch".
            The model is not trained for a number of iterations
            given by `epochs`, but merely until the epoch
            of index `epochs` is reached.
        '''
        return model.fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps)
    @classmethod
    def train_on_batch(self,model,x, y, sample_weight=None, class_weight=None):
        '''
        手动将一个个batch的数据送入网络中训练
        Runs a single gradient update on a single batch of data.
        '''
        return model.train_on_batch(x, y, sample_weight, class_weight)
    @classmethod
    def evaluate(self,model,x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None):
        '''
        Returns the loss value & metrics values for the model in test mode.

        Computation is done in batches.
        '''
        return model.evaluate(x, y, batch_size, verbose, sample_weight, steps)
    @classmethod
    def predict(self,model,x, batch_size=None, verbose=0, steps=None):
        '''
        Generates output predictions for the input samples.

        Computation is done in batches.
        '''
        return model.predict(x, batch_size, verbose, steps)
        