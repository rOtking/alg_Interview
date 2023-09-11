import numpy as np
import cv2





path = '/Users/zhaobo/Desktop/IMG_4689.JPG'

a = np.array([[1,2,9],[4,5,6],[1,3,9]], dtype=np.float)

b = np.array([])
b[:,2] = b[:,2] + 1
print(b)