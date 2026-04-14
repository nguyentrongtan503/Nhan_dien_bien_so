import math
import cv2
import numpy as np
import Preprocess

# Constants
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
Min_char = 0.01
Max_char = 0.09

# Load Image
img = cv2.imread("data/image/19.jpg")
img = cv2.resize(img, dsize=(1920, 1080))

# Preprocessing (Contrast Enhancement)
imgGrayscaleplate, imgThreshplate, imgTopHat, imgBlackHat = Preprocess.preprocess(img)
imgGrayscalePlusTopHatMinusBlackHat, imgTopHat, imgBlackHat = Preprocess.maximizeContrast(imgGrayscaleplate)
imgThreshplate = cv2.adaptiveThreshold(imgGrayscaleplate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

cv2.imshow("Thresholded Plate", imgThreshplate)

# Load KNN Model
npaClassifications = np.loadtxt("classifications.txt", np.float32)
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))  # Reshape to 1D
kNearest = cv2.ml.KNearest_create()
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

# Image Preprocessing
canny_image = cv2.Canny(imgThreshplate, 250, 255)  # Canny Edge Detection
kernel = np.ones((3, 3), np.uint8)
dilated_image = cv2.dilate(canny_image, kernel, iterations=1)  # Dilation
cv2.imshow("Dilated Image", dilated_image)

# Find contours for license plate
contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image = cv2.cvtColor(dilated_image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
cv2.imshow("Contour Image", contour_image)

# Sort and filter contours to identify the plate
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Top 10 largest contours
screenCnt = []

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.06 * peri, True)
    [x, y, w, h] = cv2.boundingRect(approx.copy())
    ratio = w / h

    if len(approx) == 4:
        screenCnt.append(approx)
        cv2.putText(img, str(len(approx.copy())), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

if not screenCnt:
    print("No plate detected")
else:
    for screenCnt in screenCnt:
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

        # Find angle of the license plate
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = screenCnt.reshape(4, 2)
        array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        array.sort(reverse=True, key=lambda x: x[1])
        (x1, y1), (x2, y2) = array[0], array[1]
        doi = abs(y1 - y2)
        ke = abs(x1 - x2)
        angle = math.atan(doi / ke) * (180.0 / math.pi)

        # Crop and align the plate
        mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
        x, y = np.where(mask == 255)
        (topx, topy), (bottomx, bottomy) = (np.min(x), np.min(y)), (np.max(x), np.max(y))

        roi = img[topx:bottomx, topy:bottomy]
        imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
        ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

        if x1 < x2:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
        else:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

        roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
        imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))
        roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
        imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

        # Character segmentation
        kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
        cont, _ = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imshow("Morphed Image", thre_mor)
        cv2.drawContours(roi, cont, -1, (100, 255, 255), 2)

        # Character recognition
        char_x_ind = {}
        char_x = []
        height, width, _ = roi.shape
        roiarea = height * width

        for ind, cnt in enumerate(cont):
            (x, y, w, h) = cv2.boundingRect(cont[ind])
            ratiochar = w / h
            char_area = w * h

            if Min_char * roiarea < char_area < Max_char * roiarea and 0.25 < ratiochar < 0.7:
                if x in char_x:
                    x += 1
                char_x.append(x)
                char_x_ind[x] = ind

        char_x = sorted(char_x)
        first_line = ""
        second_line = ""

        for i in char_x:
            (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
            cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)

            imgROI = thre_mor[y:y + h, x:x + w]
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
            npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
            npaROIResized = np.float32(npaROIResized)
            _, npaResults, _, _ = kNearest.findNearest(npaROIResized, k=3)
            strCurrentChar = str(chr(int(npaResults[0][0])))

            if y < height / 3:
                first_line += strCurrentChar
            else:
                second_line += strCurrentChar

        print(f"\nLicense Plate: {first_line} - {second_line}\n")
        roi = cv2.resize(roi, None, fx=0.75, fy=0.75)
        cv2.imshow(f"License Plate {n}", cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

# Resize and display the final image
img = cv2.resize(img, None, fx=0.5, fy=0.5)
cv2.imshow('License Plate Detection', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
