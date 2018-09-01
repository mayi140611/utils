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
    def Variable(self, initial_value, trainable=True, collections=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None, expected_shape=None, import_scope=None, constraint=None):
        '''
        Create a variable.
        After construction, the type and shape of
        the variable are fixed. The value can be changed using one of the assign
        methods.
        If you want to change the shape of a variable later you have to use an
        `assign` Op with `validate_shape=False`.
        '''
        return tf.Variable(initial_value, trainable, collections, validate_shape, caching_device, name, variable_def, dtype, expected_shape, import_scope, constraint)
    @classmethod
    def placeholder(self, dtype, shape=None, name=None):
        '''
        占位符
        Inserts a placeholder for a tensor that will be always fed.
        @shape: 占位符的形状。
        [None, 1]：表示行数不确定，1列
        **Important**: This tensor will produce an error if evaluated. Its value must
        be fed using the `feed_dict` optional argument to `Session.run()`,
        `Tensor.eval()`, or `Operation.run()`.
        ```python
        x = tf.placeholder(tf.float32, shape=(1024, 1024))
        y = tf.matmul(x, x)

        with tf.Session() as sess:
          print(sess.run(y))  # ERROR: will fail because x was not fed.

          rand_array = np.random.rand(1024, 1024)
          print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
        ```
        '''
        return tf.placeholder(dtype, shape=None, name=None)
    '''
    #####################################
    生成Tensor
    #####################################
    '''
    @classmethod
    def zeros(self, shape, dtype=tf.float32, name=None):
        '''
        Creates a tensor with all elements set to zero.

        '''
        return tf.zeros(shape, dtype, name)
    @classmethod
    def random_normal(self, shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None):
        '''
        Outputs random values from a normal distribution.
        '''
        return tf.random_normal(shape, mean, stddev, dtype, seed, name)
    
    '''
    #####################################
    op操作
    注意：tf中定义的操作都是lazy_load，在session中才会进行实际的计算
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
    def equal(self, x, y, name=None):
        '''
        Returns the truth value of (x == y) element-wise.
        '''
        return tf.equal(x, y, name)
    @classmethod
    def argmax(self, input, axis=None, name=None, dimension=None, output_type=tf.int64):
        '''
        Returns the index with the largest value across axes of a tensor. (deprecated arguments)
        '''
        return tf.argmax(input, axis, name, dimension, output_type)
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
    @classmethod
    def matmul(self, a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, name=None):
        '''
        矩阵乘法
        Multiplies matrix `a` by matrix `b`, producing `a` * `b`.
        '''
        return tf.matmul(a, b, transpose_a, transpose_b, adjoint_a, adjoint_b, a_is_sparse, b_is_sparse, name)
    '''
    #####################################
    op操作: 激活函数
    注意：tf中定义的操作都是lazy_load，在session中才会进行实际的计算
    #####################################
    '''
    @classmethod
    def tanh(self, x, name=None):
        '''
        Computes hyperbolic tangent（双曲正切函数） of `x` element-wise.
        tanh(x)=sinh(x)/cosh(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x))
        '''
        return tf.nn.tanh(x, name)
    @classmethod
    def softmax(self, logits, axis=None, name=None, dim=None):
        '''
        Computes softmax activations. (deprecated arguments)
        softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
        '''
        return tf.nn.softmax(logits, axis=None, name=None, dim=None)
    @classmethod
    def relu(self, features, name=None):
        '''
        Computes rectified linear: `max(features, 0)`.
        '''
        return tf.nn.relu(features, name)
    
    '''
    #####################################
    op操作: 损失函数loss function
    常用的损失函数有：
    * 二次代价函数，均方误差
    适用于激活函数是线性（如Relu）的
    loss = tf.reduce_mean(tf.square(y-prediction))
    * 交叉熵损失函数
    如果输出层使用了Sigmoid函数后直接输出（没有使用softmax）
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=prediction))
    #如果输出层采用了softmax
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction))
    #####################################
    '''
    @classmethod
    def mse(self, y, prediction):
        '''
        Mean Square Error
        @y: 真实值
        @prediction: 预测值
        '''
        return tf.reduce_mean(tf.square(y-prediction))
    @classmethod
    def sigmoid_cross_entropy_with_logits(self, y, prediction):
        '''
        交叉熵损失函数，如果输出层使用了Sigmoid函数后直接输出（没有使用softmax）
        @y: 真实值
        @prediction: 预测值
        '''
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=prediction))
    @classmethod
    def softmax_cross_entropy_with_logits_v2(self, y, prediction):
        '''
        交叉熵损失函数，如果输出层采用了softmax
        @y: 真实值
        @prediction: 预测值
        '''
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction))
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
    def RMSPropOptimizer(self, learning_rate=0.001, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, centered=False, name='RMSProp'):
        '''
        Optimizer that implements the RMSProp algorithm.
        注意学习率太大会严重影响准确率，一般去0.001
        '''
        return tf.train.RMSPropOptimizer(learning_rate, decay, momentum, epsilon, use_locking, centered, name)
    @classmethod
    def AdamOptimizer(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam'):
        '''
        Optimizer that implements the Adam algorithm.
        注意学习率太大会严重影响准确率，一般去0.001
        '''
        return tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon, use_locking, name)
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
    '''
    #####################################
    layer
    #####################################
    '''
    def Dense(self):
        '''
        全连接层
        这里只是一个例子
        '''
        Weights_L1 = tf.Variable(tf.random_normal([1,10]))#[1,10] = [输入神经元个数, 输出神经元个数]
        # Weights_L1 = tf.Variable(np.random.normal(size=[1,10]))#报错，类型不匹配
        biases_L1 = tf.Variable(tf.zeros([1,10]))
        Wx_plus_b_L1 = tf.matmul(x,Weights_L1) + biases_L1
        L1 = tf.nn.tanh(Wx_plus_b_L1)        
    def dropout(self, x, keep_prob, noise_shape=None, seed=None, name=None):
        '''
        Computes dropout.
        dropout。本质也是减小模型的拟合能力（复杂程度）。
        在训练时人为的随机的关闭一些神经元不参与训练；
        注意：测试和真正的使用时则是打开所有的神经元的。
        一般应用在Dence层后，如
        L1 = tf.nn.tanh(tf.matmul(x,W1)+b1)
        L1_drop = tf.nn.dropout(L1,keep_prob) 
        '''
        return tf.nn.dropout(x, keep_prob, noise_shape, seed, name)
    