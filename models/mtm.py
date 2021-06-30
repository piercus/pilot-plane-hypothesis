from tensorflow.keras import backend as K
from tensorflow.keras import layers

class MTM(layers.Layer):
  def __init__(
        self,
        shape,
        BlockClass,
        depths,
        input_shape,
        name="MTM",
        **kwargs
    ):
    super(MTM, self).__init__(name=name, **kwargs)
    self.depths = depths
    self.blocks = [BlockClass(depth=depth, input_shape=input_shape) for depth in self.depths]
    
    self.dropout = layers.Dropout(rate=(len(depths)-1)/len(depths), noise_shape=(batch_size,)+(1,1,)+(self.depths,), seed=None, **kwargs)
    self.concat = layers.Concatenate(axis=-1)
    self.expand_dim = layers.Reshape(input_shape+(1,))
    self.sum = layers.Lambda(lambda x: K.sum(
        x, axis=-1, keepdims=False
    ))

  def call(self, inputs):
    block_out = [self.expand_dim(block(inputs)) for block in self.blocks]
    self.dropout(self.concat(block_out))
    self.sum(block_out)
    return block_out