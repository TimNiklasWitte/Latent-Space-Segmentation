#!/bin/bash


#python3 ./Autoencoder/Training.py
#python3 

for i in {1..25}
do
    cd Autoencoder
    python3 ./Training.py
    cd ..
    python3 Visualize_segmentation.py --path "./results/$i.png"
done
