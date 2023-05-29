import tensorflow as tf
import tensorflow.compat.v1 as tfcompat

tfcompat.disable_v2_behavior()


def dense_layer(inputs, output_units, bias=True, activation=None, batch_norm=None,
                dropout=None, scope='dense-layer', reuse=False):
    """
    Applies a dense layer to a 2D tensor of shape [batch_size, input_units]
    to produce a tensor of shape [batch_size, output_units].
    Args:
        inputs: Tensor of shape [batch size, input_units].
        output_units: Number of output units.
        activation: activation function.
        dropout: dropout keep prob.
    Returns:
        Tensor of shape [batch size, output_units].
    """
    with tfcompat.variable_scope(scope, reuse=reuse):
        W = tfcompat.get_variable(
            name='weights',
            initializer=tfcompat.keras.initializers.VarianceScaling(scale=2.0),
            shape=[shape(inputs, -1), output_units]
        )
        z = tf.matmul(inputs, W)
        if bias:
            b = tfcompat.get_variable(
                name='biases',
                initializer=tfcompat.constant_initializer(),
                shape=[output_units]
            )
            z = z + b

        if batch_norm is not None:
            z = tfcompat.layers.batch_normalization(z, training=batch_norm, reuse=reuse)

        z = activation(z) if activation else z
        z = tf.nn.dropout(z, rate=1 - (dropout)) if dropout is not None else z
        return z


def time_distributed_dense_layer(
        inputs, output_units, bias=True, activation=None, batch_norm=None,
        dropout=None, scope='time-distributed-dense-layer', reuse=False):
    """
    Applies a shared dense layer to each timestep of a tensor of shape
    [batch_size, max_seq_len, input_units] to produce a tensor of shape
    [batch_size, max_seq_len, output_units].

    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        output_units: Number of output units.
        activation: activation function.
        dropout: dropout keep prob.

    Returns:
        Tensor of shape [batch size, max sequence length, output_units].
    """
    with tfcompat.variable_scope(scope, reuse=reuse):
        W = tfcompat.get_variable(
            name='weights',
            initializer=tfcompat.keras.initializers.VarianceScaling(scale=2.0),
            shape=[shape(inputs, -1), output_units]
        )
        z = tf.einsum('ijk,kl->ijl', inputs, W)
        if bias:
            b = tfcompat.get_variable(
                name='biases',
                initializer=tfcompat.constant_initializer(),
                shape=[output_units]
            )
            z = z + b

        if batch_norm is not None:
            z = tfcompat.layers.batch_normalization(z, training=batch_norm, reuse=reuse)

        z = activation(z) if activation else z
        z = tf.nn.dropout(z, rate=1 - (dropout)) if dropout is not None else z
        return z


def shape(tensor, dim=None):
    """Get tensor shape/dimension as list/int"""
    if dim is None:
        return tensor.shape.as_list()
    else:
        return tensor.shape.as_list()[dim]


def rank(tensor):
    """Get tensor rank as python list"""
    return len(tensor.shape.as_list())
