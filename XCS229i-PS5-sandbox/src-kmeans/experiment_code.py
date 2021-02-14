from matplotlib.image import imread
import matplotlib.pyplot as plt

A = imread("peppers-small.tiff")
plt.imshow(A)
plt.show()
A_ = A.reshape(-1,3)
A_red = A[:,:,0]
A_green = A[:,:,1]
A_blue = A[:,:,2]
