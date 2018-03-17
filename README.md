[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Deep Learning Project ##

In this project, the objective is to train a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry. Utilizing "semantic segmentation" of images captured by a quadrotor drone and "fully convoluted networks" to identify various parts of the image as backgroudn or general person or specific persion ("hero") this objective is enabled. Tensorflow was used to setup the model and was run on a GPU on my desktop.

[image_0]: ./docs/misc/sim_screenshot.png
![alt text][image_0] 

## Fully Convoluted Neural Network Architecture ##
The neural network is implemented in the 'model_training.ipynb' file using Tensorflow and Keras. The model consists of an input layer followed by 3 encoder layers, a middle layer (1x1 convolution), 3 decoder layers and an output layer as shown in the image below. The model is implemented in ```python def fcn_model(input, num_classes):``` function.

[image_1]: ./docs/misc/fcnn.png
![alt_text][image_1]

**Input Layer**
This is the image being processed.

**Encoder Layers**
These layers are convolutional layers that reduce to a deeper 1x1 conv layer. Since these layers preserve spatial information, they are useful in sematic segmentation of images. We use batch normalization in these layers.

**Middle Layer**
The middle layer is a 1x1 conv layer with a kernel size of 1 and stride of 1. We use batch normalization in this layer.

**Decoder Layers**
These layers upsample encoded layers back to a higher dimension. The method used for upsampling is called `bilinear upsampling` which is a technique using weighted average of 4 nearest pizels located diagonally to a given pizel to estimate new pixel intesity value. We use batch normalization in these layers.

**Skip connections**
We introduce skip connections between some encoder and decoder layers to improve the resolution of the result.

## Training, Predicting and Scoring ##
### Hyperparameters ###
The modle has the following hyperparameters that can be tuned to get a good prediction and to prevent overfitting while training.

**Optimzer Algorithm**
Due to the large size of the dataset, it makes sense to use a mini-batch gradient descent method. I used `Adam` optimizer algorithm which uses `momentum` to speed up convergence. This is also an `Adaptive Gradient Algorithm` that maintains a per-parameter learning rate. 

**Learning rate**
In order to atain a fast yet stable convergence an optimal `learning rate`. If a small learning rate is used, it slows down convergence and/or gets stuck on a local minimum. Too fast of a learning rate may cause instabilities. I tried learning rates of `0.001` and `0.004`, found better performance at `0.004`.

**Batch size**
In my experiments, I tried a number of batch sizes ranging from `50` to `25`, and the best results were obtained for a batch size of `30`.

**Number of epochs**
This is the number of times the algorithm sees the entire dataset. Choosing a large number will lead to overfitting. But choosing a small number will lead to underfitting. I found `number of epochs = 40` gave me a decent result.

**Stepse per epoch**
Steps per epoch is simply the size of the training dataset divided by batch size. I used the default `200` as the steps.

**Validation steps per epoch**

In order to be able to use all validation data available, validation steps per epoch is simply the size of the validation dataset divided by batch size. I used the default `50` as the validation steps per epoch. 

**Number of workers**

This refers to the number of parallel threads that will be spawned to train the model. I used the `2` and `4` and saw no significant improvements with `4` so stuck to `2`.

### Parameter tuning
In this project, parameters were tuned by brute force until a good enough model was obtained.

**Scoring**

To score the network on the Follow Me task, two types of error are measured. First the intersection over the union for the pixelwise classifications is computed for the target channel. Intersection over union is a useful metric for semantic segmentation tasks. It is the ratio between the area of overlap between the prediction and the ground truth, and the area of union. Perfect prediction will lead to an IoU of 1. 

In addition to this we determine whether the network detected the target person or not. If more then 3 pixels have probability greater then 0.5 of being the target person then this counts as the network guessing the target is in the image. 

We determine whether the target is actually in the image by whether there are more then 3 pixels containing the target in the label mask. 

Using the above the number of detection true_positives, false positives, false negatives are counted. 

The final score is the pixelwise `average_IoU*(n_true_positive/(n_true_positive+n_false_positive+n_false_negative))` on data similar to that provided in sample_evaulation_data. The following table shows the ious and final score of my model.
| Learning rate |	Batch size |	Optimizer |	number of epochs	| Steps per epoch |	validation steps per epoch |	iou1o | iou1h |	iou2o |	iou2h |	iou3o |	iou3h	| finalIOU |	finalScore |
|0.004 |	30 |	Adam |	40 |	200 |	50 |	0.29 |	0.81 |	0.71 |	0 |	0.38 |	0.11 |	0.46 |	0.33 |

The following images show the sample image take by the drone, the labelled ground truth and the prediction of the model. The hero is labelled in blue and others are in green. Red is background.

**Hero in the image**
[image_2]: ./docs/misc/follow_target1.png
![alt_text][image2]
[image_3]: ./docs/misc/follow_target2.png
![alt_text][image3]
[image_4]: ./docs/misc/follow_target3.png
![alt_text][image3]

**No hero in image**
[image_5]: ./docs/misc/follow_nontarget1.png
![alt_text][image5]
[image_6]: ./docs/misc/follow_nontarget2.png
![alt_text][image6]
[image_7]: ./docs/misc/follow_nontarget3.png
![alt_text][image7]

**Hero at distance**
[image_8]: ./docs/misc/follow_targetdistance1.png
![alt_text][image8]
[image_9]: ./docs/misc/follow_targetdistance2.png
![alt_text][image9]
[image_10]: ./docs/misc/follow_targetdistance3.png
![alt_text][image10]

## Future Improvements ##
*1. Model architecture and hyperparameters can be tuned to improve the overall score.
*2. Implement data augmentation techniques to increas the data.
