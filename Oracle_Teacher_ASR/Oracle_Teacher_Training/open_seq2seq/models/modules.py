# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.
Building blocks for Transformer
'''

import numpy as np
import tensorflow as tf


def ln(inputs, epsilon=1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs



def get_token_embeddings(vocab_size, num_units, zero_pad=True):
    '''Constructs token embedding matrix.
    Note that the column of index 0's are set to zeros.
    vocab_size: scalar. V.
    num_units: embedding dimensionalty. E.
    zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
    To apply query/key masks easily, zero pad is turned on.
    Returns
    weight variable: (V, E)
    '''
    with tf.variable_scope("shared_weight_matrix"):
        embeddings = tf.get_variable('weight_mat',
                                     dtype=tf.float32,
                                     shape=(vocab_size, num_units),
                                     initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, num_units]),
                                    embeddings[1:, :]), 0)
    return embeddings



def scaled_dot_product_attention(Q, K, V, key_masks, causality=False, dropout_rate=0., training=True, scope="scaled_dot_product_attention"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]
        
        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)
        
        # scale 
        outputs /= d_k ** 0.5
        
        # key masking
        outputs = mask(outputs, key_masks=key_masks, type="key")


        # causality or future blinding masking
        if causality: 
            outputs = mask(outputs, type="future")
        
        # softmax 
        outputs = tf.nn.softmax(outputs)
        temp_attention = outputs 
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))
       
        # dropout 
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
        
        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs, temp_attention



def mask(inputs, key_masks=None, type=None):
    """Masks paddings on keys or queries to inputs
    inputs: 3d tensor. (h*N, T_q, T_k)
    key_masks: 3d tensor. (N, 1, T_k)
    type: string. "key" | "future"
    e.g.,
    >> inputs = tf.zeros([2, 2, 3], dtype=tf.float32)
    >> key_masks = tf.constant([[0., 0., 1.],
                                [0., 1., 1.]])
    >> mask(inputs, key_masks=key_masks, type="key")
    array([[[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],
       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]],
       [[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],
       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]]], dtype=float32)
    """
    padding_num = -2 ** 15 +1#-2 ** 16 + 1
    if type in ("k", "key", "keys"):
        key_masks = tf.to_float(key_masks)
        key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1])  # (h*N, seqlen)
        key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)
        outputs = inputs + tf.cast(key_masks,tf.float16) * tf.cast(padding_num,tf.float16)
    if type in ("q", "query", "queries"):
        key_masks = tf.to_float(key_masks)
        key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1])
        key_masks = tf.expand_dims(key_masks, 1)
        outputs = inputs + tf.transpose(key_masks,[0,2,1]) * padding_num #- tmp*padding_num

    return outputs

        
def multihead_attention(queries, keys, values, key_masks, 
                        num_heads=8,
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):

    d_model = queries.get_shape().as_list()[-1]
    batch_size = queries.get_shape().as_list()[0]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections

        Q = tf.layers.dense(queries, d_model, use_bias=False, trainable=training)  # (N, T_q, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=False, trainable=training)  # (N, T_k, d_model)
        V = tf.layers.dense(values, d_model, use_bias=False, trainable=training)  # (N, T_k, d_model)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        # Attention
        outputs, attention = scaled_dot_product_attention(Q_, K_, V_, key_masks, causality, dropout_rate, training)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)

        outputs += queries
        outputs = ln(outputs)

    return outputs


def conv_elems_1d(x, factor, out_depth=None):
  """Decrease the length and change the dimensionality.
  Merge/restore/compress factors positions of dim depth of the input into
  a single position of dim out_depth.
  This is basically just a strided convolution without overlap
  between each strides. The original length has to be divided by factor.
  Args:
    x (tf.Tensor): shape [batch_size, length, depth]
    factor (int): Length compression factor.
    out_depth (int): Output depth
  Returns:
    tf.Tensor: shape [batch_size, length//factor, out_depth]
  """
  out_depth = out_depth or x.get_shape().as_list()[-1]
  # with tf.control_dependencies(  # Dynamic assertion
  #     [tf.assert_equal(tf.shape(x)[1] % factor, 0)]):
  x = tf.expand_dims(x, 1)  # [batch_size, 1, length, depth]
  x = tf.layers.conv2d(inputs=x,
      name="conv_elems_2d",
      filters=out_depth,
      kernel_size=(1, factor),
      strides=(1, factor),
      padding="valid",
      data_format="channels_last",)  # [batch_size, 1, length//factor, out_depth]
  x = tf.squeeze(x, 1)  # [batch_size, length//factor, depth]
  return x


def ff(inputs, training, num_units, scope="positionwise_feedforward"):
    '''position-wise feed forward net. See 3.3

    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        #inputs = ln(inputs)
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1], activation=tf.nn.relu)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = ln(outputs)
    return outputs


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    V = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / V)


def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.
    returns
    3d tensor that has the same shape as inputs.
    '''

    E = inputs.get_shape().as_list()[-1]  # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float16)#tf.float32)  # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)



def split_heads(x, hidden_size, num_heads):
    """Split x into different heads, and transpose the resulting value.
    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.
    Args:
      x: A tensor with shape [batch_size, length, hidden_size]
    Returns:
      A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
    """
    with tf.name_scope("split_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[1]

      # Calculate depth of last dimension after it has been split.
      depth = (hidden_size // num_heads)

      # Split the last dimension
      x = tf.reshape(x, [batch_size, length, num_heads, depth])

      # Transpose the result
      return tf.transpose(x, [0, 2, 1, 3])

def combine_heads(x, hidden_size):
    """Combine tensor that has been split.
    Args:
      x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]
    Returns:
      A tensor with shape [batch_size, length, hidden_size]
    """
    with tf.name_scope("combine_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[2]
      x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
      return tf.reshape(x, [batch_size, length, hidden_size])

def attention_mask(key_masks):
    with tf.name_scope("attention_bias"):
        key_masks = tf.to_float(key_masks)
        attention_mask = key_masks * -1e9 #float16=-1e4
        attention_mask = tf.expand_dims(tf.expand_dims(attention_mask,1),1)
    return attention_mask

def layer_norm(x, hidden_size):
    epsilon = 0.0001
    scale = tf.get_variable("layer_norm_scale", [hidden_size],initializer=tf.keras.initializers.Ones(),dtype=tf.float32)
    bias = tf.get_variable("layer_norm_bias", [hidden_size],initializer=tf.keras.initializers.Zeros(),dtype=tf.float32)
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    x = x - mean
    variance = tf.reduce_mean(tf.abs(x), axis=[-1], keepdims=True)
    norm_x = tf.div(x , variance + epsilon)
    y = norm_x * scale + bias
    return y
