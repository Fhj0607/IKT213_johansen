import cv2
import numpy as np

# define global variables
img = cv2.imread('./images/iris.jpg')
height, width, channels = img.shape
ePA = np.zeros((height, width, channels), dtype=np.uint8) # ePA short for emptyPictureArray

def padding(image, border_width): # A & B
    padded_img = cv2.copyMakeBorder(
        image,
        border_width,border_width,border_width,border_width,
        cv2.BORDER_REFLECT
    ) # C
    cv2.imwrite('./images/padded.jpg', padded_img) # D
    return padded_img
padding(img, 100) # C define border width value 100

def crop(image, x_0, x_1, y_0, y_1): # A & B
    cropped_img = image[y_0:y_1, x_0:x_1] # C
    cv2.imwrite('./images/cropped.jpg', cropped_img) # D
    return cropped_img
crop(img, 200, -130, 200, -130) # C define x_0, x_1, y_0 and y_1 values

def resize(image, w, h): # A & B: I use w and h to avoid conflicts with my global variables. In practice, it fulfills the task specification correctly.
    resized_img = cv2.resize(image, (w, h)) # C
    cv2.imwrite('./images/resized.jpg', resized_img) # D
    return resized_img
resize(img,200,200) # C define w and h values 200

def copy(image): # A & B
    # loops through img values
    for y in range(height):
        for x in range(width):
            ePA[y, x] = image[y, x] # C

    cv2.imwrite('./images/manual_copy.jpg', ePA) # D
    return ePA
copy(img)

def grayscale(image): # A & B
    grayed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # C
    cv2.imwrite('./images/grayed.jpg', grayed_image) # D
    return grayed_image
grayscale(img)

def hsv(image): # A & B
    hsved_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # C
    cv2.imwrite('./images/hsved.jpg', hsved_image) # D
    return hsved_image
hsv(img)

def hue_shifted(image, emptyPictureArray, hue): # A & B
    # loops through all img values
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                value = int(image[y, x, c]) + hue

                # accounts for "out of bounds" color values for uint8
                # C sub-task ?
                if value > 255:
                    value = 255
                if value < 0:
                    value = 0

                emptyPictureArray[y, x, c] = value # C

    cv2.imwrite('./images/hue_shifted.jpg', emptyPictureArray) # D
    return emptyPictureArray
hue_shifted(img, ePA, 50) # C: define color value 50

def smoothing(image): # A & B
    ksize = (15,15) # C
    blurred_image = cv2.GaussianBlur(image, ksize, sigmaX=0, borderType=cv2.BORDER_DEFAULT) # C
    cv2.imwrite('./images/smoothed.jpg', blurred_image) # D
    return blurred_image
smoothing(img)

def rotation(image, rotation_angle): # A & B
    if rotation_angle == 90 : rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) # C
    if rotation_angle == 180: rotated_image = cv2.rotate(image, cv2.ROTATE_180) # D
    cv2.imwrite('./images/rotated.jpg', rotated_image)
    return rotated_image
rotation(img, 180) # E define rotation angle 180








