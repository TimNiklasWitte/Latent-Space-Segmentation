import sys
sys.path.append("./Autoencoder")
sys.path.append("./MNIST Model")

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse

from Autoencoder import *
from MNIST_Model import *

from matplotlib.patches import Patch

def main():

    parser = argparse.ArgumentParser(description="Visualize the latent space segmentation")
    parser.add_argument("--path", help="Path where the latent space segmentation map (image) shall be stored.", required=True)
    parser.add_argument("--show", help="Display the latent space segmentation map", default=False, required=False)
    args = parser.parse_args()

    path = args.path
    show = args.show 

    autoencoder = Autoencoder()
    autoencoder.build(input_shape=(None, 32, 32, 1))
    autoencoder.load_weights("./Autoencoder/saved_models/trained_weights_20").expect_partial()
   
    mnist_model = MNIST_Model()
    mnist_model.build(input_shape=(None, 32, 32, 1))
    mnist_model.load_weights("./MNIST Model/saved_models/trained_weights_20").expect_partial()

    num_points = 200
    coordinates = np.linspace(-1, 1, num_points)
    
    embedding_list = []
    for x in coordinates:
        for y in coordinates:
            embedding_list.append([x,y])
    
    embeddings = tf.convert_to_tensor(embedding_list)
    
    imgs = autoencoder.decoder(embeddings)
   
    preds = mnist_model(imgs)
    preds = tf.argmax(preds, axis=-1)

    labels = tf.reshape(preds, shape=(num_points, num_points))
    print(labels)
    visualize_segmentation(labels, path, show)


def visualize_segmentation(labels, path=None, show=False):
    
    colors = np.array([
        [0,0,0], # black
        [128, 128, 128], # white
        [255, 0, 0], # red
        [0, 255, 0], # green
        [0, 0, 255], # blue
        [255, 255, 0], # yellow
        [255, 0, 255], # magenta
        [0, 255, 255], # cyan
        [128, 0, 0], # brown
        [128, 128, 0], # brown
    ])

    latent_space_segmentation = colors[labels] 
    plt.imshow(latent_space_segmentation, extent =[-1,1,-1, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    
    # Legend: Add colors manually
    ax = plt.legend().axes
    handles, labels = ax.get_legend_handles_labels()

    patches_list = []
    for label, color in enumerate(colors):
        color = color/255
        handles.append(Patch(color=color))
        labels.append(label)

        patch = Patch(color=color, label=label)
        patches_list.append(patch)

    lgd = ax.legend(handles=patches_list, loc='lower right', bbox_to_anchor=(1.24, 0.3))

    plt.tight_layout()

    if path:
        plt.savefig(path)
    
    if show:
        plt.show()
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")