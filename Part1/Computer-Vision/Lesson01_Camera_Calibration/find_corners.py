import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from networkx.algorithms.distance_measures import radius

# prepare object points
nx = 8 #TODO: 输入一行里Corner的个数
ny = 6 #TODO: 输入一列里Corner的个数

# Make a list of calibration images
fname = './IGNORE/calibration_test.png'
img = cv2.imread(fname)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

corners_4 = np.concatenate((corners[0:2,:,:],corners[8:10,:,:]))

# If found, draw corners
if ret == True:
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

#[[[]],[[]]]
    for i in corners_4:
        x,y = i[0]
        cv2.circle(img,(int(x),int(y)),10,(0,0,0),-1)


    plt.imshow(img)
    plt.show()