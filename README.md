ResNet50 is a powerful model for image classification when it is trained for an adequate number of iterations.

Training set: 1080 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (180 pictures per number).

Test set: 120 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (20 pictures per number).

Here are examples for each number, and corresponding labels converted to one-hot. 
![alt signs_dataset](https://raw.githubusercontent.com/tejaslodaya/tensorflow-signs-nn/master/signs_dataset.png)

Architecture:
1. Input is an image of size 64 x 64 x 3 (RGB), which is normalized by dividing 255
2. Model: 
    ![alt architecture](https://raw.githubusercontent.com/tejaslodaya/keras-signs-resnet/master/images/resnet_kiank.png?token=AKA30X6IfVA3fA1B2LPt4zp3ld_djoX5ks5aE7KYwA%3D%3D)
    which comprises of identity block:
    ![alt identity](https://raw.githubusercontent.com/tejaslodaya/keras-signs-resnet/master/images/idblock3_kiank.png?token=AKA30Rbs8_ILc-eDskqQc3vYWjMtCDUwks5aE7RcwA%3D%3D)
    and convolution block:
    ![alt convolution](https://raw.githubusercontent.com/tejaslodaya/keras-signs-resnet/master/images/convblock_kiank.png?token=AKA30Q7Z31keHP48mqRsjLojaaaTzgnyks5aE7SFwA%3D%3D)
3. The last fully connected layer gives a probability of the image belonging to one of the six classes.
4. RELU activation function. Categorical cross entropy cost. Adam optimizer
5. Mini-batch gradient descent with minibatch_size of 32

The model is CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK x 2 -> CONVBLOCK -> IDBLOCK x 3
-> CONVBLOCK -> IDBLOCK x 5 -> CONVBLOCK -> IDBLOCK x 2 -> AVGPOOL -> FLATTEN -> FC

Outcome:

1.  Test Accuracy - 0.866
    Loss - 0.53
2.  TODO- to overcome overfitting, add L2 or dropout regularization