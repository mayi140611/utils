#!/usr/bin/python
# encoding: utf-8
# version: tensorflow==1.9.0


import tensorflow as tf
class tf_wrapper(object):
    def __init__(self):
        pass
    @classmethod
    def constant(self,value, dtype=None, shape=None, name='Const', verify_shape=False):
        '''
        Creates a constant tensor.
        @value: The argument `value` can be a constant value, or a list of values of type
        `dtype`. If `value` is a list, then the length of the list must be less
        than or equal to the number of elements implied by the `shape` argument (if
        specified). In the case where the list length is less than the number of
        elements specified by `shape`, the last element in the list will be used
        to fill the remaining entries.
        @shape: tuple or list. Optional dimensions of resulting tensor.
        ```python
        # Constant 1-D Tensor populated with value list.
        tensor = tf.constant([1, 2, 3, 4, 5, 6, 7]) => [1 2 3 4 5 6 7]

        # Constant 2-D tensor populated with scalar value -1.
        tensor = tf.constant(-1.0, shape=[2, 3]) => [[-1. -1. -1.]
                                                     [-1. -1. -1.]]
        ```
        '''
        return tf.constant(value, dtype, shape, name, verify_shape)
    @classmethod
    def Variable(self, initial_value=None, trainable=True, collections=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None, expected_shape=None, import_scope=None, constraint=None):
        '''
        Create a variable.
        After construction, the type and shape of
        the variable are fixed. The value can be changed using one of the assign
        methods.
        If you want to change the shape of a variable later you have to use an
        `assign` Op with `validate_shape=False`.
        '''
        return tf.Variable(initial_value, trainable, collections, validate_shape, caching_device, name, variable_def, dtype, expected_shape, import_scope, constraint)
    '''
    #####################################
    op操作
    #####################################
    '''
    @classmethod
    def assign(self, ref, value, validate_shape=None, use_locking=None, name=None):
        '''
        Update 'ref' by assigning 'value' to it.

        This operation outputs a Tensor that holds the new value of 'ref' after
          the value has been assigned. This makes it easier to chain operations
          that need to use the reset value.
        '''
        return tf.assign(ref, value, validate_shape, use_locking, name)
    @classmethod
    def add(self, x, y, name=None):
        '''
        Returns x + y element-wise.
          x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`, `string`.
          y: A `Tensor`. Must have the same type as `x`.
          name: A name for the operation (optional).

        Returns:
          A `Tensor`. Has the same type as `x`.
        '''
        return tf.add(x, y, name)
    @classmethod
    def multiply(self, x, y, name=None):
        '''
        Returns x * y element-wise.
        '''
        return multiply(x, y, name)
    @classmethod
    def square(self, x, name=None):
        '''
        Computes square of x element-wise.
        '''
        return tf.square(x, name)
    @classmethod
    def reduce_mean(self, input_tensor, axis=None, keepdims=None, name=None, reduction_indices=None, keep_dims=None):
        '''
        Computes the mean of elements across dimensions of a tensor. (deprecated arguments)
        @keepdims: If true, retains reduced dimensions with length 1.
        ```python
        x = tf.constant([[1., 1.], [2., 2.]])
        tf.reduce_mean(x)  # 1.5
        tf.reduce_mean(x, 0)  # [1.5, 1.5]
        tf.reduce_mean(x, 1)  # [1.,  2.]
        ```
        '''
        return tf.reduce_mean(input_tensor, axis=None, keepdims=None, name=None, reduction_indices=None, keep_dims=None)
    
    '''
    #####################################
    session操作
    #####################################
    '''
    @classmethod
    def Session(self, target='', graph=None, config=None):
        '''        
        Creates a new TensorFlow session. A class for running TensorFlow operations.

        If no `graph` argument is specified when constructing the session,
        the default graph will be launched in the session. If you are
        using more than one graph (created with `tf.Graph()` in the same
        process, you will have to use different sessions for each graph,
        but each graph can be used in multiple sessions. In this case, it
        is often clearer to pass the graph to be launched explicitly to
        the session constructor.
        '''
        return tf.Session(target='', graph=None, config=None)
    @classmethod
    def run(self, sess, fetches, feed_dict=None, options=None, run_metadata=None):
        '''        
        Runs operations and evaluates tensors in `fetches`.
    
        This method runs one "step" of TensorFlow computation, by
        running the necessary graph fragment(片段) to execute every `Operation`
        and evaluate every `Tensor` in `fetches`, substituting the values in
        `feed_dict` for the corresponding input values.

        The `fetches` argument may be a single graph element, or an arbitrarily
        nested list, tuple, namedtuple, dict, or OrderedDict containing graph
        elements at its leaves. 
        @fetches: 可以是graph中的任意op
        '''
        return sess.run(fetches, feed_dict, options, run_metadata)
    @classmethod
    def global_variables_initializer(self):
        '''
        Returns an Op that initializes global variables.
        '''
        return tf.global_variables_initializer()
    '''
    #####################################
    train操作
    #####################################
    '''
    @classmethod
    def GradientDescentOptimizer(self, learning_rate, use_locking=False, name='GradientDescent'):
        '''
        Optimizer that implements the gradient descent algorithm.
        '''
        return tf.train.GradientDescentOptimizer(learning_rate, use_locking, name)
    @classmethod
    def minimize(self, optimizer, loss, global_step=None, var_list=None, gate_gradients=1, aggregation_method=None, colocate_gradients_with_ops=False, name=None, grad_loss=None):
        '''
        Add operations to minimize `loss` by updating `var_list`.

        This method simply combines calls `compute_gradients()` and
        `apply_gradients()`. If you want to process the gradient before applying
        them call `compute_gradients()` and `apply_gradients()` explicitly instead
        of using this function.
        '''
        return optimizer.minimize(loss, global_step, var_list, gate_gradients, aggregation_method, colocate_gradients_with_ops, name, grad_loss)