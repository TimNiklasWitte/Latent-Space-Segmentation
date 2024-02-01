import tensorflow as tf

from Encoder import *
from Decoder import *

class Autoencoder(tf.keras.Model):
    def __init__(self):

        super(Autoencoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.loss_function = tf.keras.losses.MeanSquaredError()

        self.metric_loss = tf.keras.metrics.Mean(name="loss")

    @tf.function
    def call(self, x):

        embedding = self.encoder(x)
        x_reconstructed = self.decoder(embedding)
         
        return x_reconstructed
    
    @tf.function
    def train_step(self, x):
  
        with tf.GradientTape() as tape:
            x_reconstructed = self(x)
            loss = self.loss_function(x, x_reconstructed)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metric_loss.update_state(loss)

        return loss
    

    @tf.function
    def test_step(self, dataset):
          
        self.metric_loss.reset_states()

        for x in dataset:
            x_reconstructed = self(x)

            loss = self.loss_function(x, x_reconstructed)
            self.metric_loss.update_state(loss)
    