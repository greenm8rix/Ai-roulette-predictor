import tensorflow as tf

# -----------------------------
# Attention Layer Implementation
# -----------------------------
class Attention(tf.keras.layers.Layer):
    """
    Custom Attention layer for sequence processing.

    This layer computes attention weights for input sequences and
    produces a context vector by taking a weighted sum of the inputs.
    """
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.W = None
        self.b = None

    def build(self, input_shape):
        """
        Creates the weights of the layer.

        Args:
            input_shape: The shape of the input tensor.
        """
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], input_shape[-1]),
            initializer="random_normal",
            trainable=True
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True
        )
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        """
        Performs the forward pass of the layer.

        Args:
            inputs: The input tensor, shape (batch, time_steps, features).

        Returns:
            A context vector, shape (batch, features).
        """
        # inputs shape: (batch, time_steps, features)
        # score shape: (batch, time_steps, features)
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)

        # attention_weights shape: (batch, time_steps, features)
        # Note: Softmax is applied across the time_steps dimension (axis=1)
        # to get weights for each time step.
        attention_weights = tf.nn.softmax(score, axis=1)

        # context shape: (batch, time_steps, features)
        context = attention_weights * inputs

        # context_vector shape: (batch, features)
        # Summing over the time_steps dimension to get the context vector.
        context_vector = tf.reduce_sum(context, axis=1)

        return context_vector

    def get_config(self):
        """Returns the config of the layer."""
        config = super(Attention, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        """Creates a layer from its config."""
        return cls(**config)
