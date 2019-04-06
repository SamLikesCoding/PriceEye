"""

    PricEye Backend App example
    Created by Sarath Peter

"""
import predictor_keras_model as PricEye
import os.path

inp_img = input(" Enter input image path : ")
inst = PricEye.price_model()
inst.load_dataset()
inst.data_augment()
inst.model_loader("Model/priceye_keras.model")
if(os.path.exists("./"+inp_img)):
	result = inst.predictor("./"+inp_img)
	print(" The product is : {}".format(result))
else:
	print("ERROR : File Not found!")
