"""

    PricEye     : Keras Based Product Predict Model
    Build       : V1
    Created By  : Sarath Peter

"""

'''    Required Libraries    '''

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.models import load_model
from keras.models import Model
from shutil import copy2
from glob import glob
import numpy as np
import os

'''    
    PlaidML support - PLaidML is a library for Machine Learning to use Non - Nvidia GPUs
    PlaidML also helps to work on Nvidia platforms but CuDA is best for that 

'''
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
'''
        The Class Definition for model, 
        The whole class represents Neural Network module, 
        
        Whole process:
        
        Loading Dataset -> Creating Model(downloads the model if not found) -> Data Augmentation
                                                                                     |
                                                                                     V
                                    Train the Model using given Dataset <-(No)- Is Model trained?
                                                |                                     |
                                                |                                   (Yes)
                                                |                                     |
                                                V                                     V
                                    Prediction on input image <----------- Load the Trained Model
                                                |
                                                V
                                        Result of Prediction
                                        
'''

class price_model:

    # Initialisation of parameters and other required stuff
    def __init__(self):
        self.num_classes = 0  # Count number of classes
        self.train_classes = [] # Class Labels
        self.load_vector = []  # Pointer list to dataset classpaths
        self.load_object = []  # Pointer list to each image in dataset
        self.train_sets = []  # List of images used for training
        self.test_sets = []  # List of images used for testing
        self.model = None  # The model of Neural Network
        self.trainGen = None  # Training Data Generator (feeds Training Data)
        self.validGen = None  # Testing and Validation Data Generator (feeds Testing and Validation Data)
        self.WIDTH = 299  # Width of input images
        self.HEIGHT = 299  # Height of input images
        self.BATCH_SIZE = 32  # Size of batch inputs feed to Neural Network
        self.EPOCHS = 5  # Epoch parameter
        self.STEPS_PER_EPOCH = 320  # Number of steps for each epoch
        self.VALIDATION_STEPS = 64  # Number of steps for Validation
        # Saves the Model as for future uses
        self.MODEL_FILE = 'Model/priceye_keras.model'
        # self.MODEL_FILE = 'Model/priceye_keras.h5' used for TensorFlow experiments

    # Sets Image Dimensions, Default Size = 299x299
    def set_img_dim(self, H, W):
        print("--- Setting Image Dimensions ---")
        self.HEIGHT = H
        self.WIDTH = W

    # Sets Parameters for model
    def set_param(self, epoch, batch_size, epoch_steps, validation_steps):
        print("--- Setting Epoch Parameters ---")
        self.EPOCHS = epoch
        self.BATCH_SIZE = batch_size
        self.STEPS_PER_EPOCH = epoch_steps
        self.VALIDATION_STEPS = validation_steps

    # Loading the Dataset
    def load_dataset(self):
        print("\n\n--- Getting input train_set ---")
        tmp1 = []
        tmp2 = []
        for path, dir, filenames in os.walk("train_set"):
            self.load_vector.append(path)  # Lists out paths in Dataset
        self.load_vector = self.load_vector[1:]  # Removes root path (because it's not useful)
        self.num_classes = len(self.load_vector)  # Gets count of classes in dataset
        for imageset in self.load_vector:
            self.load_object.append(glob(imageset + "/*.jpeg"))  # Gets all image paths from all classes
        self.load_vector = [x.split('/')[1] for x in self.load_vector]  # Lambda function to remove root path
        for images in self.load_object:
            tmp1, tmp2 = train_test_split(images, test_size=0.30)  # Splitting images for training and testing
            self.train_sets.append(tmp1)  # Train Images Here
            self.test_sets.append(tmp2)  # Test Images Here
        print(" Datasets loaded : {}".format(self.load_vector))
        print(" No of classes : {} \n".format(self.num_classes))
        for i in range(self.num_classes):
            for test_img in self.test_sets[i]:
                # Seperates Testing Images to test directory
                if not os.path.exists('test_set/' + str(self.load_vector[i])):
                    os.mkdir('test_set/' + str(self.load_vector[i]))
                print("{} -> test_set".format(test_img))
                copy2(test_img, 'test_set/' + str(self.load_vector[i]) + "/")  #
            print(" {} Dataset have {} train images and {} test images"
                  .format(self.load_vector[i],
                          len(self.train_sets[i]),
                          len(self.test_sets[i])))

    # Creates the Model
    def model_creator(self):
        print("\n\n--- Creating model ---")
        '''
        
        The Neural Network used here is Inception Version 3, Inception is an ImageNet used to train images over not edges
        but also ability to retrain images. Inception model will be downloaded if not found in running system, or else it 
        will be found at /.keras folder in user's root (not system root) directory
        
        '''

        base_model = InceptionV3(weights='imagenet', include_top=False)  # Gets the model
        tmp = base_model.output  # Sets the Output layer
        tmp = GlobalAveragePooling2D(name='avg_pool')(tmp)  # Sets Global Average Pooling layer
        tmp = Dropout(0.4)(tmp)  # Sets Dropout layer
        pred = Dense(self.num_classes, activation='softmax')(tmp)  # Sets Dense Softmax Layer
        self.model = Model(inputs=base_model.input, outputs=pred)  # All Layers are connected to Net
        for layer in base_model.layers:
            layer.trainable = False  # Setting layers as static
        print("Building model....")
        self.model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',  # Finally the Neural Network Model is born!!
            metrics=['accuracy']
        )

    # Does the Image preprocessing
    def data_augment(self):
        print("\n\n--- Data Augmentation ---")
        '''
        
        Data Generators are used to preprocess images before training and testing 
        as part of optimizing network learning.
        
        '''

        # Data Generator for Training Dataset
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        # Data Generator for Testing Dataset, testing data is re-split to validation data at here
        validation_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        # Connecting Training Data generator
        self.trainGen = train_datagen.flow_from_directory(
            'train_set',
            target_size=(self.HEIGHT, self.WIDTH),
            batch_size=self.BATCH_SIZE,
            class_mode='categorical'
        )

        self.train_classes = self.trainGen.class_indices

        # Connecting Testing Data Generator
        self.validGen = validation_datagen.flow_from_directory(
            'test_set',
            target_size=(self.HEIGHT, self.WIDTH),
            batch_size=self.BATCH_SIZE,
            class_mode='categorical'
        )

    # The Transfer Learning Function
    def transfer_learn(self):
        print("\n\n--- Transfer Learning ---")

        # The Model learns images at here
        history = self.model.fit_generator(
            self.trainGen,
            epochs=self.EPOCHS,
            steps_per_epoch=self.STEPS_PER_EPOCH,
            validation_data=self.validGen,
            validation_steps=self.VALIDATION_STEPS)

        # to reuse the model in future it's saved in Model Directory
        self.model.save(self.MODEL_FILE)

    # Predicts the class
    def predictor(self, img):
        print("\n\n--- Custom Predictor Online ---")
        p_img = image.load_img(img, target_size=(self.HEIGHT, self.WIDTH))  # Loads the image to be predicted
        x = image.img_to_array(p_img)  # Converts image to numpy array
        x = np.expand_dims(x, axis=0)  # Expands Image array
        x = preprocess_input(x)  # Preprocess image before prediction
        pred = self.model.predict(x)  # Predicts the class
        # return pred[0]  # returns the result as numpy ndarray
        return list(self.train_classes.keys())[pred.argmax(axis=1)[0]] # Returns the class label

    # If you want another model to be used, Use this
    def model_loader(self, model_path):
        print("--- Loading Model ---")
        self.model = load_model(model_path)
        # What's inside model
        self.model.summary()
        # To know the shape of model
        # for layer in self.model.layers:
        #   print("Shape : {}".format(layer.output_shape))

# FOR TESTING PURPOSE ONLY
if __name__ == '__main__':
    inst = price_model()
    inst.load_dataset()
    inst.data_augment()
    # inst.model_creator()
    # inst.transfer_learn() # If the model needs to be trained
    inst.model_loader("Model/priceye_keras.model")
    result = inst.predictor('test_case1.jpeg')
    print(" The Model found object as {} ".format(result))
    # Old Method
    # print(list(result))
    # print("The model predicts the as {}".format(inst.load_vector[list(result).index(max(result))]))
    