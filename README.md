Dislocation Detection in Grain Boundaries

This repository contains an implementation of dislocation probability estimation and localization within grain boundaries using CNN architectures The model is designed to predict the probability of dislocation presence and provide their (x, y) positions in given images.

The CNN is designed with 4 convolutional layers followed by batch normalization and maximum pooling. It has two output heads:

Classification: prediction of the probability of the presence of a dislocation.
Regression: prediction of the (x,y) coordinates of dislocations.

CNN key points:
 Compact architecture with 4 convolutional layers.
 Binary classification of the presence of dislocations.
 Regression of (x,y) coordinates.


Results :
           The model is fast and gives 100% accuracy for classification and 92% accuracy for regression.
