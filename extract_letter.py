import cv2 as cv
import os
import imutils

input_image = 'generated_captcha_images'
output_image = 'extracted_letter_images'

input_file = os.listdir(input_image)
count = {}
for i, image_name in enumerate(input_file):
    correct_letter = image_name.split('.')[0]
    print("Processing the verification image {}/{}:".format(i, len(input_file)))
    image = cv.imread(input_image+'/'+image_name)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.copyMakeBorder(image, 8, 8, 8, 8, cv.BORDER_REPLICATE)
    image = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)[1]

    contours = cv.findContours(image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]
    letter_region = []
    #Deal with the contours of one image
    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)
        if w/h > 1.25:
            half_width = int(w/2)
            letter_region.append((x, y, half_width, h))
            letter_region.append((x + half_width, y, half_width, h))
        else:
            letter_region.append((x,y,w,h))

    if len(letter_region) != 4:
        continue
    letter_image_regions = sorted(letter_region, key=lambda x: x[0])
    for region, letter_single in zip(letter_image_regions, correct_letter):
        x, y, w, h = region
        output_region = image[y - 2:y + 2 + h, x - 2: x + 2 + w]
        count_name = count.get(letter_single, 1)
        file_path = output_image + '/' + str(letter_single) + str(count_name) + '.png'
        cv.imwrite(file_path, output_region)
        count[letter_single] = count_name + 1