import cv2
import os
import numpy as np

# loop semua gambar
for filename in os.listdir("gambar"):
    if filename.endswith(".jpg"):
        img_path = os.path.join("gambar", filename)

        # Read Gambar
        img = cv2.imread(img_path)

        # Thresholding
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

        # Grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Brightening
        bright_img = cv2.convertScaleAbs(img, beta=100)

        # operasi aritmatik
        img1 = cv2.imread('gambar/image1.jpg')
        img2 = cv2.imread('gambar/image2.jpg')
        add = cv2.add(img1, img2)
        subtract = cv2.subtract(img1, img2)
        multiply = cv2.multiply(img1, img2)
        divide = cv2.divide(img1, img2)

        # operasi boolean
        for i in range(1, 2):
            img1 = cv2.imread('gambar/image' + str(i) + '.jpg')
            img2 = cv2.imread('gambar/image' + str(i+1) + '.jpg')
            and_op = cv2.bitwise_and(img1, img2)
            or_op = cv2.bitwise_or(img1, img2)
            xor_op = cv2.bitwise_xor(img1, img2)
            not_op = cv2.bitwise_not(img1)

        # Geometri
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
        rot_img = cv2.warpAffine(img, M, (cols, rows))
        # Sobel Edge Detection
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        # Canny Edge Detection
        canny = cv2.Canny(img, 100, 200)
        # Gaussian Blur
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        gaussian = cv2.Canny(blur, 100, 200)

        # output gambar
        cv2.imshow('Original', img)
        cv2.imshow('Thresholding', thresh_img)
        cv2.imshow('Grayscale', img_gray)
        cv2.imshow('Brightening', bright_img)
        # Operasi Aritmatik
        cv2.imshow('Addition', add)
        cv2.imshow('Subtraction', subtract)
        cv2.imshow('Multiplication', multiply)
        cv2.imshow('Division', divide)
        # Operasi Boolean 
        cv2.imshow('AND', and_op)
        cv2.imshow('OR', or_op)
        cv2.imshow('XOR', xor_op)
        cv2.imshow('NOT', not_op)
        # Operasi Geometri
        cv2.imshow('Geometri', rot_img)

        # Edge Detection
        cv2.imshow('Gausian Blur Edge Detection',+ gaussian)
        cv2.imshow('Sobel Edge Detection',+ sobel)
        cv2.imshow('Canny Edge Detection',+ canny)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
