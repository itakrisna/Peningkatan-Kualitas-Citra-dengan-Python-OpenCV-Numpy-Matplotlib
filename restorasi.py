import cv2
import numpy as np
import matplotlib.pyplot as plt

# Membaca citra
image = cv2.imread('malam.jpeg', cv2.IMREAD_GRAYSCALE)

# Negative Transformation
negative_image = 255 - image

# Log Transformation
c = 255 / np.log(1 + np.max(image))
log_image = c * (np.log(image + 1))

# Contrast Stretching
a = 0
b = 255
c_min = np.min(image)
c_max = np.max(image)
contrast_stretched_image = (image - c_min) * ((b - a) / (c_max - c_min)) + a

# Bit Plane Slice
def get_bit_plane(image, bit):
    return (image >> bit) & 1

bit_plane = 7  # Change this value (0 to 7) to select the desired bit plane
bit_plane_image = get_bit_plane(image, bit_plane) * 255

# Histogram Equalization
hist_equalized_image = cv2.equalizeHist(image)

# Noise Removal (Sharpening)
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # Sharpening kernel
sharpened_image = cv2.filter2D(image, -1, kernel)

# Menampilkan hasil
cv2.imshow('Original Image', image)
cv2.imshow('Negative Transformation', negative_image)
cv2.imshow('Log Transformation', log_image.astype(np.uint8))
cv2.imshow('Contrast Stretching', contrast_stretched_image.astype(np.uint8))
cv2.imshow(f'Bit Plane Slice {bit_plane}', bit_plane_image)
cv2.imshow('Histogram Equalization', hist_equalized_image)
cv2.imshow('Sharpened Image', sharpened_image)

# Plot histograms
fig, axs = plt.subplots(2, 4, figsize=(15, 10))

axs[0, 0].hist(image.ravel(), bins=256, range=[0, 256], color='r', alpha=0.5)
axs[0, 0].set_title('Original Image Histogram')

axs[0, 1].hist(negative_image.ravel(), bins=256, range=[0, 256], color='g', alpha=0.5)
axs[0, 1].set_title('Negative Transformation Histogram')

axs[0, 2].hist(log_image.ravel(), bins=256, range=[0, 256], color='b', alpha=0.5)
axs[0, 2].set_title('Log Transformation Histogram')

axs[0, 3].hist(contrast_stretched_image.ravel(), bins=256, range=[0, 256], color='c', alpha=0.5)
axs[0, 3].set_title('Contrast Stretching Histogram')

axs[1, 0].hist(bit_plane_image.ravel(), bins=256, range=[0, 256], color='m', alpha=0.5)
axs[1, 0].set_title(f'Bit Plane Slice {bit_plane} Histogram')

axs[1, 1].hist(hist_equalized_image.ravel(), bins=256, range=[0, 256], color='y', alpha=0.5)
axs[1, 1].set_title('Histogram Equalization Histogram')

axs[1, 2].hist(sharpened_image.ravel(), bins=256, range=[0, 256], color='k', alpha=0.5)
axs[1, 2].set_title('Sharpened Image Histogram')

# Remove axis labels
for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

    # Membaca citra
image = cv2.imread('malam.jpeg', cv2.IMREAD_GRAYSCALE)

# Histogram Equalization
hist_equalized_image = cv2.equalizeHist(image)

# Menampilkan hasil
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Original Image
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

# Histogram Original Image
axs[0, 1].hist(image.ravel(), bins=256, range=[0, 256], color='r', alpha=0.5)
axs[0, 1].set_title('Original Image Histogram')

# Histogram Equalized Image
axs[1, 1].hist(hist_equalized_image.ravel(), bins=256, range=[0, 256], color='g', alpha=0.5)
axs[1, 1].set_title('Histogram Equalized Image')

# Equalized Image
axs[1, 0].imshow(hist_equalized_image, cmap='gray')
axs[1, 0].set_title('Equalized Image')
axs[1, 0].axis('off')

# Remove axis labels
for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
