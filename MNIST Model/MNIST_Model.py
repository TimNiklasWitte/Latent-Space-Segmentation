import tensorflow as tf

class MNIST_Model(tf.keras.Model):

    def __init__(self):
        super(MNIST_Model, self).__init__()

        self.layer_list = [
            tf.keras.layers.Conv2D(8, kernel_size=(3, 3), strides=(2,2), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(2,2), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(2,2), padding='same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(25, activation="tanh"),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ]

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.loss_function = tf.keras.losses.BinaryCrossentropy()

        self.metric_loss = tf.keras.metrics.Mean(name="loss")
        self.metric_accuracy = tf.keras.metrics.Accuracy(name="accuracy")


    @tf.function
    def call(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x

    @tf.function
    def train_step(self, x, target):

        with tf.GradientTape() as tape:
            prediction = self(x)
            loss = self.loss_function(target, prediction)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metric_loss.update_state(loss)

        prediction = tf.argmax(prediction, axis=-1)
        label = tf.argmax(target, axis=-1)
        self.metric_accuracy.update_state(label, prediction)

    @tf.function
    def test_step(self, dataset):
          
        self.metric_loss.reset_states()
        self.metric_accuracy.reset_states()

        for x, target in dataset:
            prediction = self(x)

            loss = self.loss_function(target, prediction)
            self.metric_loss.update_state(loss)

            prediction = tf.argmax(prediction, axis=-1)
            label = tf.argmax(target, axis=-1)
            self.metric_accuracy.update_state(label, prediction)
