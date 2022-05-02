import numpy as np
import cv2

def erode(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    kernel_ones_count = kernel.sum()
    eroded_img = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] - 1))
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            if kernel_ones_count == (kernel * img[i:i_, j:j_]).sum() / 255:
                eroded_img[i + kernel_center[0], j + kernel_center[1]] = 1

    return eroded_img[:img_shape[0], :img_shape[1]]


'''
TODO: implement morphological operators
'''
def dilate(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    kernel_ones_count = kernel.sum()
    dilate_img = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] - 1))
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            if (kernel * img[i:i_, j:j_]).sum() / 255 >= 1:
                dilate_img[i + kernel_center[0], j + kernel_center[1]] = 1
    return dilate_img[:img_shape[0], :img_shape[1]]

def opening(img, kernel):
    img_erosion = erode(img, kernel)
    #Do lúc erosion với dilation return về thì giá trị ảnh là từ 0 - 1 nên phải * 255
    img_erosion = img_erosion*255
    img_opening = dilate(img_erosion, kernel)
    return img_opening

def closing(img, kernel):
    img_dilation = dilate(img, kernel)
    img_closing = erode(img_dilation, kernel)
    return img_closing

def Hit_or_Miss(img):
    # Tạo 2 ma trận kernel hit và miss
    kernel_HIT = np.array((
        [0, 0, 0],
        [1, 1, 0],
        [0, 1, 0]), dtype="int")
    kernel_MISS = np.array((
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0]), dtype="int")
    #Ma trận ảnh nhị phân ngược
    inverted_image = cv2.bitwise_not(img)

    HIT_image= erode(img,kernel_HIT)
    MISS_image= erode(inverted_image,kernel_MISS)

    HoM_image= HIT_image * MISS_image
    return HoM_image

def BoundaryExtraction(img, kernel):
    img_erosion = erode(img, kernel)*255

    BE_image =img-img_erosion

    return BE_image

def Thinning(img, kernel):
    img_HoM = Hit_or_Miss(img)*255
    print(img_HoM)
    Thinning_image =img-img_HoM

    return Thinning_image