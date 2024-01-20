import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras .preprocessing.image import ImageDataGenerator

#descomprimir zip
#dataset_path ="./cats_and_dogs_filtered.zip"
#zip_object = zipfile.ZipFile(file=dataset_path, mode="r")
#zip_object.extractall("./")
#zip_object.close()

#path al set de datos
dataset_path_new = "./cats_and_dogs_filtered"

train_dir= os.path.join(dataset_path_new, "train")
validation_dir= os.path.join(dataset_path_new, "validation")
#El modelo
IMG_SHAPE = (128,128,3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,include_top = False,weights="imagenet")
#base_model.summary()
#congelamos
base_model.trainable = True

#REDUCIR TAMAÑO
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
#CAPA DE SALIDA
prediction_layer = tf.keras.layers.Dense(units=1, activation ='sigmoid')(global_average_layer)

#CRearmodelo
model= tf.keras.models.Model(inputs=base_model.input, outputs=prediction_layer)
#compilar
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),loss="binary_crossentropy",metrics=["accuracy"])

#data GEnerators
data_gen_train = ImageDataGenerator(rescale=1/255.)
data_gen_valid = ImageDataGenerator(rescale=1/255.)

#asignar valores a generators
train_generator= data_gen_train.flow_from_directory(train_dir, target_size=(128,128),batch_size=128, class_mode="binary")
valid_generator= data_gen_valid.flow_from_directory(validation_dir, target_size=(128,128),batch_size=128, class_mode="binary")

#Entrenar
model.fit(train_generator, epochs=5, validation_data=valid_generator)

#Evaluar la eficacia del modelo
valid_loss, valid_accuracy =model.evaluate(valid_generator)
#afinamiento
fine_tune_at=100

#aplicado a capas
for layer in base_model.layers[:fine_tune_at:]:
    layer.trainable= False

#compilado
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),loss="binary_crossentropy",metrics=["accuracy"])

# #entrenar model
model.fit(train_generator,epochs= 5, validation_data=valid_generator)

batch_size = 128
img_size = 128
test_dir = './cats_and_dogs_filtered/test'
img_gen = ImageDataGenerator(rescale= 1/255)

test_img_gen = img_gen.flow_from_directory(batch_size =batch_size,
                                               directory=test_dir,
                                               shuffle= False,
                                               target_size=(img_size, img_size),
                                               class_mode=None)


test_total = len(os.listdir(test_dir))
pred = model.predict_generator(test_img_gen, steps=1, verbose =1)

predicted_class_indices = np.round(pred)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k[0]] for k in predicted_class_indices]
filenames = test_img_gen.filenames

results = pd.DataFrame({"Filename":filenames,
                        "Predictions":predictions})
print(results)
