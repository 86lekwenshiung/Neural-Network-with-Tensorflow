# Tensorflow Developer Certification Learning Journey

#### Sources and Credits
___
1. Zero to Mastery Deep Learning with TensorFlow course
2. https://github.com/mrdbourke/tensorflow-deep-learning
3. [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
4. [Neural Network Playground](https://playground.tensorflow.org/)

## Content Table
<a name = 'main_table'></a>
| No. | Notebook Dir | Key Summary | Data |
|---|---|---|---|
| 01 | [TF Regression](https://github.com/86lekwenshiung/Neural-Network-with-Tensorflow/blob/main/01_Neural_Network_Regression_With_Tensorflow.ipynb) | [Typical Acrhitecture for Regression](#tf_regression) | - |
| 02 | [TF Classification](https://github.com/86lekwenshiung/Neural-Network-with-Tensorflow/blob/main/02_Neural_Network_Classification_With_Tensorflow.ipynb) | [Typical Acrhitecture for Classification](#tf_classification) | - |
| 03 | [TF CNN](https://github.com/86lekwenshiung/Neural-Network-with-Tensorflow/blob/main/04_Transfer Learning with Tensorflow Part 1..ipynb) | [Typical Acrhitecture for CNN](#tf_cnn) | 10_food_classes_all_data |
| 04 | [TF Transfer Learning : Feature Extraction](https://github.com/86lekwenshiung/Neural-Network-with-Tensorflow/blob/main/03_Convolutional_Neural_Network_(CNN)_With_Tensorflow_.ipynb) | [Transfer Learning Feature Extraction](#tf_transfer_learning) |  |

<a name = 'tf_regression'></a>
### 1.0 Tensorflow Regression Basic Architecture
[(back to top)](#main_table)

| **Hyperparameter** | **Typical value** |
| --- | --- |
| Input layer shape | Same shape as number of features (e.g. 3 for # bedrooms, # bathrooms, # car spaces in housing price prediction) |
| Hidden layer(s) | Problem specific, minimum = 1, maximum = unlimited |
| Neurons per hidden layer | Problem specific, generally 10 to 100 |
| Output layer shape | Same shape as desired prediction shape (e.g. 1 for house price) |
| Hidden activation | Usually [ReLU](https://www.kaggle.com/dansbecker/rectified-linear-units-relu-in-deep-learning) (rectified linear unit) |
| Output activation | None, ReLU, logistic/tanh |
| Loss function | [MSE](https://en.wikipedia.org/wiki/Mean_squared_error) (mean square error) or [MAE](https://en.wikipedia.org/wiki/Mean_absolute_error) (mean absolute error)/Huber (combination of MAE/MSE) if outliers |
| Optimizer | [SGD](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD) (stochastic gradient descent), [Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) |

***Table 1:*** *Typical architecture of a regression network.* ***Source:*** *Adapted from page 293 of [Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow Book by Aurélien Géron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)*
___

<a name = 'tf_classification'></a>
### 2.0 Tensorflow Classification Basic Architecture
[(back to top)](#main_table)

| **Hyperparameter** | **Binary Classification** | **Multiclass classification** |
| --- | --- | --- |
| Input layer shape | Same as number of features (e.g. 5 for age, sex, height, weight, smoking status in heart disease prediction) | Same as binary classification |
| Hidden layer(s) | Problem specific, minimum = 1, maximum = unlimited | Same as binary classification |
| Neurons per hidden layer | Problem specific, generally 10 to 100 | Same as binary classification |
| Output layer shape | 1 (one class or the other) | 1 per class (e.g. 3 for food, person or dog photo) |
| Hidden activation | Usually [ReLU](https://www.kaggle.com/dansbecker/rectified-linear-units-relu-in-deep-learning) (rectified linear unit) | Same as binary classification |
| Output activation | [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) | [Softmax](https://en.wikipedia.org/wiki/Softmax_function) |
| Loss function | [Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression) ([`tf.keras.losses.BinaryCrossentropy`](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy) in TensorFlow) | Cross entropy ([`tf.keras.losses.CategoricalCrossentropy`](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy) in TensorFlow) |
| Optimizer | [SGD](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD) (stochastic gradient descent), [Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) | Same as binary classification |

***Table 1:*** *Typical architecture of a classification network.* ***Source:*** *Adapted from page 295 of [Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow Book by Aurélien Géron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)*
___


<a name = 'tf_cnn'></a>
### 3.0 Tensorflow CNN Basic Architecture
[(back to top)](#main_table)

| **Hyperparameter/Layer type** | **What does it do?** | **Typical values** |
| ----- | ----- | ----- |
| Input image(s) | Target images you'd like to discover patterns in| Whatever you can take a photo (or video) of |
| Input layer | Takes in target images and preprocesses them for further layers | `input_shape = [batch_size, image_height, image_width, color_channels]` |
| Convolution layer | Extracts/learns the most important features from target images | Multiple, can create with [`tf.keras.layers.ConvXD`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D) (X can be multiple values) |
| Hidden activation | Adds non-linearity to learned features (non-straight lines) | Usually ReLU ([`tf.keras.activations.relu`](https://www.tensorflow.org/api_docs/python/tf/keras/activations/relu)) |
| Pooling layer | Reduces the dimensionality of learned image features | Average ([`tf.keras.layers.AvgPool2D`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D)) or Max ([`tf.keras.layers.MaxPool2D`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D)) |
| Fully connected layer | Further refines learned features from convolution layers | [`tf.keras.layers.Dense`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) |
| Output layer | Takes learned features and outputs them in shape of target labels | `output_shape = [number_of_classes]` (e.g. 3 for pizza, steak or sushi)|
| Output activation | Adds non-linearities to output layer | [`tf.keras.activations.sigmoid`](https://www.tensorflow.org/api_docs/python/tf/keras/activations/sigmoid) (binary classification) or [`tf.keras.activations.softmax`](https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax) |

|Hyperparameter Name|Description|Typical Values|
|---|---|---|
|Filter|How many filters should pass over an input tensors|10,32,64,128 (higher value , more complex|
|Kernel Size(filter size)| Shape of the filter over the output| 3,5,7, lower value = smaller features. Higher value = larger features|
|Padding|Pad the target sensor with 0s at the border(if 'same') to preserve input shape. Or leaves in the target sensor(if 'valid') , lowering output shape|'same or 'valid'|
|Strides| No. of steps a filter takes across an image at a time(if stride = 1 , a filter moves across an image 1 pixel at a time| 1(default) ,2|
___

<a name = 'tf_transfer_learning'></a>
### 4.0 Tensorflow Feature Extraction Overview
[(back to top)](#main_table)

1. **Transfer learning** is when you take a pretrained model as it is and apply it to your task without any changes. 
 
    * For example, many computer vision models are pretrained on the ImageNet dataset which contains 1000 different classes of images. This means passing a single image to this model will produce 1000 different prediction probability values (1 for each class). 
    * This is helpful if you have 1000 classes of image you'd like to classify and they're all the same as the ImageNet classes, however, it's not helpful if you want to classify only a small subset of classes (such as 10 different kinds of food). Model's with `"/classification"` in their name on TensorFlow Hub provide this kind of functionality.

2. **Feature extraction transfer learning** is when you take the underlying patterns (also called weights) a pretrained model has learned and adjust its outputs to be more suited to your problem. 

    * For example, say the pretrained model you were using had 236 different layers (EfficientNetB0 has 236 layers), but the top layer outputs 1000 classes because it was pretrained on ImageNet. To adjust this to your own problem, you might remove the original activation layer and replace it with your own but with the right number of output classes. 
    * The important part here is that **only the top few layers become trainable, the rest remain frozen**. This way all the underlying patterns remain in the rest of the layers and you can utilise them for your own problem. This kind of transfer learning is very helpful when your data is similar to the data a model has been pretrained on.

3. **Fine-tuning transfer learning** is when you take the underlying patterns (also called weights) of a pretrained model and adjust (fine-tune) them to your own problem. 

    * This usually means training **some, many or all** of the layers in the pretrained model. This is useful when you've got a large dataset (e.g. 100+ images per class) where your data is slightly different to the data the original model was trained on. 
    * A common workflow is to "freeze" all of the learned patterns in the bottom layers of a pretrained model so they're untrainable. And then train the top 2-3 layers of so the pretrained model can adjust its outputs to your custom data (**feature extraction**). 
    * After you've trained the top 2-3 layers, you can then gradually "unfreeze" more and more layers and run the training process on your own data to further **fine-tune** the pretrained model.

<p align = 'center'>
  <img src = 'images/04-different-kinds-of-transfer-learning.png'>
 </p>
