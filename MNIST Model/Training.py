import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
import datetime

from MNIST_Model import *

NUM_EPOCHS = 10
BATCH_SIZE = 32

def main():

    #
    # Load dataset
    #   
    train_ds, test_ds = tfds.load("mnist", split=["train", "test"], as_supervised=True)

    train_dataset = train_ds.apply(prepare_data)
    test_dataset = test_ds.apply(prepare_data)
    
    #
    # Logging
    #
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = f"logs/{current_time}"
    train_summary_writer = tf.summary.create_file_writer(file_path)

    #
    # Initialize model.
    #
    model = MNIST_Model()
    model.build(input_shape=(None, 28, 28, 1))
    model.summary()

    #
    # Train and test loss/accuracy
    #
    print(f"Epoch 0")
    log(train_summary_writer, model, train_dataset, test_dataset, 0)

    #
    # Train loop
    #
    for epoch in range(1, NUM_EPOCHS + 1):
            
        print(f"Epoch {epoch}")

        for x, target in tqdm.tqdm(train_dataset, position=0, leave=True): 
            model.train_step(x, target)

        log(train_summary_writer, model, train_dataset, test_dataset, epoch)

        # Save model (its parameters)
        model.save_weights(f"./saved_models/trained_weights_{epoch}", save_format="tf")


def log(train_summary_writer, model, train_dataset, test_dataset, epoch):

    # Epoch 0 = no training steps are performed 
    # test based on train data
    # -> Determinate initial train_loss and train_accuracy
    if epoch == 0:
        model.test_step(train_dataset.take(5000))

    #
    # Train
    #
    train_loss = model.metric_loss.result()
    train_accuracy = model.metric_accuracy.result()

    model.metric_loss.reset_states()
    model.metric_accuracy.reset_states()

    #
    # Test
    #

    model.test_step(test_dataset)

    test_loss = model.metric_loss.result()
    test_accuracy = model.metric_accuracy.result()

    model.metric_loss.reset_states()
    model.metric_accuracy.reset_states()

    #
    # Write to TensorBoard
    #
    with train_summary_writer.as_default():
        tf.summary.scalar(f"train_loss", train_loss, step=epoch)
        tf.summary.scalar(f"train_accuracy", train_accuracy, step=epoch)

        tf.summary.scalar(f"test_loss", test_loss, step=epoch)
        tf.summary.scalar(f"test_accuracy", test_accuracy, step=epoch)

    #
    # Output
    #
    print(f"    train_loss: {train_loss}")
    print(f"     test_loss: {test_loss}")
    print(f"train_accuracy: {train_accuracy}")
    print(f" test_accuracy: {test_accuracy}")
 
 
def prepare_data(dataset):

    # Convert data from uint8 to float32
    dataset = dataset.map(lambda img, target: (tf.cast(img, tf.float32), target) )

    #Sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    dataset = dataset.map(lambda img, target: ( (img/128.)-1., target ) )

    # One hot target
    dataset = dataset.map(lambda img, target: (img, tf.one_hot(target, depth=10)))

    # Cache
    dataset = dataset.cache()
    
    #
    # Shuffle, batch, prefetch
    #
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")