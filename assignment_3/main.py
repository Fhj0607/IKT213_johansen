import cv2
import numpy as np

# define global variables
img = cv2.imread('./images/lambo.png')
shapes = cv2.imread('./images/shapes-1.png')
temp = cv2.imread('./images/shapes_template.jpg')


def sobel_edge_detection(image): # A & B
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur using ksize=(3,3) and SigmaX=0
    blurred = cv2.GaussianBlur(gray, (3, 3), 0) # C

    # apply Sobel edge detection using dx=1, dy=1 and ksize=1
    # I took the assignment specifications literally here. Doing sobelx (1,0) then sobely(0,1),
    # like in the example provided, is arguably more standard for general edge detection.
    # below is a demonstration of how that would be done.

    # sobelx = cv2.Sobel(src=blurred, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=1)
    # sobely = cv2.Sobel(src=blurred, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=1)
    # sobel = cv2.magnitude(sobelx, sobely) # a better result but doesn't follow the assignment completely

    sobel = cv2.Sobel(src=blurred, ddepth=cv2.CV_32F, dx=1, dy=1, ksize=1) # C follows assignment but produces a worse result

    # normalize Sobel values to an 8-bit image with values from 0 to 255
    normalized = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imshow("normalized", normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("./images/sobel_edged.png", normalized) # D
    return normalized


def canny_edge_detection(image, threshold_1, threshold_2): # A & B
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur using ksize=(3,3) and SigmaX=0
    blurred = cv2.GaussianBlur(gray, (3, 3), 0) # C

    # apply Canny edge detection using the given threshold values
    canny = cv2.Canny(image=blurred, threshold1=threshold_1, threshold2=threshold_2) # C

    cv2.imshow("canny", canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("./images/canny_edged.png", canny) # D
    return canny


def template_match(image, template): # A & B
    # convert both image and template to grayscale before template matching
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # C
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) # C

    # get template dimensions for drawing rectangles later
    h, w = template.shape[:2]

    # create similarity map between the image and template
    result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED) # C

    threshold = 0.9 # C

    # find all locations where the similarity is greater than or equal to 0.9
    locations = np.where(result >= threshold)

    # draw a red rectangle around every matched location
    for pt in zip(*locations[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1) # C

    cv2.imwrite("./images/template_match.png", image) # D
    return image


def resize(image, scale_factor: int, up_or_down: str): # A & B
    resized = image.copy()

    # move up the image pyramid scale_factor number of times
    if up_or_down == "up":
        for _ in range(scale_factor):
            resized = cv2.pyrUp(resized) # C

    # move down the image pyramid scale_factor number of times
    elif up_or_down == "down":
        for _ in range(scale_factor):
            resized = cv2.pyrDown(resized) # C
    else:
        return None

    cv2.imshow("resized", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("./images/resized.png", resized) # D
    return resized


sobel_edge_detection(img)
canny_edge_detection(img, 50, 50) # C define threshold_1 and threshold_2 values 50
template_match(shapes, temp)
resize(img, 2, "down") # C define scale_factor=2 and resize direction "down"