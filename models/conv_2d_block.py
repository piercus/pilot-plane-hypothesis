from tensorflow.keras import layers

class Conv2DBlock(layers.Layer):
  def __init__(
        depth,
        input_shape,
        kernel_size=3,
        activation='relu',
        name="ConvBlock",
        **kwargs
    ):
    super(ConvBlock, self).__init__(name=name, **kwargs)
    self.depth = depth
    self.filters = input_shape[-1]
    self.conv2d = layers.Conv2D(filters=self.filters, padding="same", kernel_size, activation)
    
  def call(self, inputs):
    x = inputs
    for i in range(self.depth):
      x = self.conv2d(x)
    
    return x
    
    