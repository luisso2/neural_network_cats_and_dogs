from keras.layers import Conv2D, Dropout, Flatten, MaxPooling2D, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image
import numpy as np

classifier = Sequential()
classifier.add(Conv2D(64,(3,3),input_shape=(64,64,1),activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(units=64,activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=64,activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',
                   metrics=['accuracy'])

generator_trng = ImageDataGenerator(rescale=1./255,rotation_range=7,
                                    horizontal_flip=True,shear_range=0.2,
                                    height_shift_range=0.07,zoom_range=0.2,)
generator_tst = ImageDataGenerator(rescale=1.255)

base_trng = generator_trng.flow_from_directory('dataset/training_set',
                                               target_size=(64,64),
                                               batch_size=32,class_mode='binary',
                                               color_mode='grayscale')
base_tst = generator_tst.flow_from_directory('dataset/test_set',target_size=(64,64),
                                             batch_size=32,class_mode='binary',
                                             color_mode='grayscale')

classifier.fit_generator(base_trng,steps_per_epoch=4000/32,epochs=10,
                         validation_data=base_tst,validation_steps=1000/32)

image_tst = image.load_img('dataset/test_set/gato/cat.3950.jpg',grayscale=True,
                           color_mode='grayscale',target_size=(64,64))
image_tst = image.img_to_array(image_tst)
image_tst /= 255
image_tst = np.expand_dims(image_tst,axis=0)
prevision = classifier.predict(image_tst)
base_trng.class_indices


