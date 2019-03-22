import numpy as np
import os
import cv2 as cv
import pickle
from helper import resize_to_fit
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.preprocessing import LabelBinarizer

#create dataset
Verification_code = os.listdir('extracted_letter_images')
dataset = []
labels = []
for vc in Verification_code:
    image = cv.imread('extracted_letter_images/' + vc)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    label = vc.split('.')[0][0]
    image = resize_to_fit(image, 20, 20)
    image = np.expand_dims(image, axis=2)
    dataset.append(image)
    labels.append(label)

#Split train set and test set
dataset = np.array(dataset, dtype=float)/255
labels = np.array(labels)
(X_train, X_test, Y_train, Y_test) = train_test_split(dataset, labels, test_size=0.25)
print(X_train.shape)

#Convert labels to one-shot
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)
with open("model_labels.dat", "wb") as f:
    pickle.dump(lb, f)

#Create model
model = Sequential()
#First convnet layer
model.add(Conv2D(20, (5,5), padding='same',input_shape=(20, 20, 1), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#second convnet layer
model.add(Conv2D(50, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
#Fully connection layer
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(32, activation='softmax'))
#compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_data=(X_test, Y_test))
#Save model
model.save('model_verification_code.hdf5')



