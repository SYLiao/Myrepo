import numpy as np
import cv2 as cv
from helper import resize_to_fit
import imutils
from keras.models import load_model
import pickle

def load_image(url):
    image = cv.imread(url)
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    image = cv.copyMakeBorder(image, 8, 8, 8, 8, cv.BORDER_REPLICATE)
    image_threshold = cv.threshold(image, 0, 255, cv.THRESH_OTSU|cv.THRESH_BINARY_INV)[1]
    contours = cv.findContours(image_threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]
    letter_region = []
    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)
        if w/h > 1.25:
            new_width = int(w/2)
            letter_region.append((x. y, new_width, h))
            letter_region.append((x+new_width, y,new_width, h))
        else:
            letter_region.append((x, y, w, h))
    if len(letter_region) != 4:
        print('Your verification code is not 4-bit code!')
        return
    letter_region = sorted(letter_region, key=lambda x:x[0])
    image_4 = []
    for i, letter in enumerate(letter_region):
        print("Output {}/4 letter's region.".format(i))
        x, y, w, h = letter
        image_4.append(image_threshold[y - 2:y + h + 2, x - 2:x + w + 2])
    return image_4

def load_model_keras(url_model, url_label, images):
    if len(images) != 4:
        print("Your verification code is wrong.")
        return
    model = load_model(url_model)
    with open(url_label, 'rb') as f:
        lb = pickle.load(f)
    prediction = []
    for image in images:
        test_image = resize_to_fit(image, 20, 20)
        test_image = np.array(test_image)/255
        test_image = np.expand_dims(test_image, axis=0)
        test_image = np.expand_dims(test_image, axis=3)
        production = model.predict(test_image)
        letter = lb.inverse_transform(production)[0]
        prediction.append(letter)
    print(prediction)

def main():
    images = load_image('test.png')
    load_model_keras('model_verification_code.hdf5','model_labels.dat', images)

if __name__ == '__main__':
    main()