## Description:


* Fashion-MNIST is a dataset of Zalando's article images.
* consisting of a training set of 60,000 examples and a test set of 10,000 examples.
* Each example is a 28x28 grayscale image, associated with a label from 10 classes.

## convert types function:


*   convert the type of each image to `float32`.
*   in next step , i make 128 batches for each train & test set.

### Data Augmentation:
* I prefer generate more image with module `ImageDataGenerator` from `tensoeflow.keras.preprocessing.image` .
> this module make rotation or horizontal-flip or zooming to create new images.

---

## Residual Block Class:

* I made a class with Residual Block name, that inheritate Form Model class that is a module imported from keras library.

* first of all , i made a convolutional layer with 64 filter that show  the **dimensionality** of the output space and use a kernel with size `(1,1)` and apply **same** padding which is the same **zero** padding which is padding with zeros evenly to the left/right or up/down of the input.

* Now add a **batch normalization** that applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.
* then apply **Relu** activation function on the outputs .

* next, I made 2 blocks like the same as before with diffrent kernel_size in second block.

* Now if the number of first conv filters(channel_in) **not equal** to the number of last conv filters(channel_out) then add another conv layer with last layer filter(channel_out).

* In the last layer i going to use `ADD()` layer to takes as input a list of all previous convolutional layers tensors, all of the same shape, and returns a single tensor, and then implement a `Relu` activation function on added outputs layer .

## ResNet50:
> Now i made a list of layers that made from 6 section .
* Firstly, I made a class which it has 2 arguments : `input_shape`and `output_dim`.
* Second of all , I inheritante from Model class that is a module imported from keras library.
* Now we create our network layers.
### STEPS:
1. Start with a conv layer that has 64 *filter* and use *kernels* with size `(7,7)` and *stride* with `(2,2)`step also apply *zero* padding.

2. Next, add normalization layer and apply **Relu** fucntion on the previous layer outputs.

3. Now , i set a `MaxPool2D` with pool_size `(3,3)`that taking the maximum value over an input window (of size defined by pool_size) for each channel of the input.

4. in the next line we add Residual blocks in deffirent numbers and diffrent dimention(filter or feature maps). 

5. In the last part , after using a `GlobalAveragePooling2D` we add two fully conected layer (dense layer) that first layer has 1000 neurons with relu activation function and i set the number of second dense layer's neurons according to the number of classes of our issue ,number 10 alos use `softmax` activation function because we handle the multiclass classification task.

## Train Model:

* I decided to use `SparseCategoricalCrossentropy` for **loss** and `Adam` **optimizer**.

* I train model for 50 epochs and my **train accuracy** is %90 and **test accuracy** is %89 , also the **train loss** eual to 0.25 and **test loss** euals to 0.29 .

> if you have a better system you can train this model for more epochs and get better results.

## Learning Curve:
<th colspan="3"><img src=".\acc.png" alt="" border='3' height='300' width='300' /></th
**you can see train and test improve well in 50 epoch .**

> we can use another optimizer like `SGD` or `RMSprop` that i commented in the cell that i write the Adam optimizer.





