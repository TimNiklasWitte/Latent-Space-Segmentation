import tensorflow as tf

class Decoder(tf.keras.Model):
    def __init__(self):

        super(Decoder, self).__init__()

        self.layer_list = [
            tf.keras.layers.Dense(2 * 2 * 40, activation='relu'),
            tf.keras.layers.Reshape((2, 2, 40)), 

            tf.keras.layers.Conv2DTranspose(40, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(32, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(16, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(8, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'),
            
            tf.keras.layers.Conv2DTranspose(1, kernel_size=(3,3), strides=(1,1), padding='same', activation='tanh'),
        ]

    @tf.function
    def call(self, x):

        for layer in self.layer_list:
            x = layer(x)
  
        return x