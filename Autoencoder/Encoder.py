import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self):

        super(Encoder, self).__init__()

        self.layer_list = [
            tf.keras.layers.Conv2D(8, kernel_size=(3, 3), strides=(2,2), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(2,2), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(2,2), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2,2), padding='same', activation='relu'),
      
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2 * 2 * 64, activation='relu'),
            tf.keras.layers.Dense(2, activation='tanh')
        ]

    @tf.function
    def call(self, x):

        for layer in self.layer_list:
            x = layer(x)
  
        return x