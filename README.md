# PricEye - Keras Model

This is my final year project i've been working on days, This documentation provides my team how to use my module
## Basic Working
The objective of this project to predict what kind of product is and state it's actual price. This module deals with
prediction of object in input image

1. Loading Dataset
2. Creating the model 
    (Downloads weights is not found, else sets and connects the layers)
3. Data Augmentation
4. If there's a trained model?
    - If Yes, Load the model
    - Else then model gets trainned using training set
5. Predicts on Input Image and returns the result 

## The Code
The module is written Python 3.7 and requires Keras, Pillow, scikit-learn and uses plaidML if needed, (I used here for experiment, I don't have Nvidia platform, So...).  
###To load dataset to code:
~~~

        predictor = price_model()
        predictor.model_creator()
~~~  
###  Next we need to create model by assigning weights and biases, Creating layers and linking them, To do that  
~~~

        predictor.model_creator()
~~~
###  For pre-process training images and augmentations
~~~

        predictor.data_augment()
~~~  
###  To do transfer learning process  
~~~

        predictor.transfer_learn()
~~~  
###  If already there's trained model, To load that. 
~~~

        predictor.model_loader(_path_to_model_) # Default = ./Model
~~~  
###  To predict on given input image using model
~~~

        result = predictor.predictor(_input_image_path)
~~~
The next set of functions to set values and parameters for model.
###  To set training - testing images at dimension HxW  
~~~

        predictor.set_img_dim(H,W)
~~~
###  To set model parameters like epoch, validation steps, and batch size
~~~

        predictor.set_param(epoch, batch_size, epoch_steps, validation_steps)
~~~


To anyone who looks in my code, any suggestion and positive opinion,  
Share by mail : sampeter027@gmail.com  

Cheers and Thanks!  
Sarath