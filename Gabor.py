# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import FileWorker as FW

# Grayscale
def BGR2GRAY(img):
    # Grayscale
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray

# Gabor Filter
def Gabor_filter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    # get half size
    d = K_size // 2

    # prepare kernel
    gabor = np.zeros((K_size, K_size), dtype=np.float32)

    # each value
    for y in range(K_size):
        for x in range(K_size):
            # distance from center
            px = x - d
            py = y - d

            # degree -> radian
            theta = angle / 180. * np.pi

            # get kernel x
            _x = np.cos(theta) * px + np.sin(theta) * py

            # get kernel y
            _y = -np.sin(theta) * px + np.cos(theta) * py

            # fill kernel
            gabor[y, x] = np.exp(-(_x**2 + Gamma**2 * _y**2) / (2 * Sigma**2)) * np.cos(2*np.pi*_x/Lambda + Psi)

    # kernel normalization
    gabor /= np.sum(np.abs(gabor))

    return gabor


# Используйте фильтр Габора, чтобы воздействовать на изображение
def Gabor_filtering(gray, K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    # get shape
    H, W = gray.shape

    # padding
    # gray = np.pad(gray, (K_size//2, K_size//2), 'edge')

    # prepare out image
    out = np.zeros((H, W), dtype=np.float32)

    # get gabor filter
    gabor = Gabor_filter(K_size=K_size, Sigma=Sigma, Gamma=Gamma, Lambda=Lambda, Psi=0, angle=angle)
    # plt.imshow(gabor)
    # plt.show()

    # filtering
    out = cv2.filter2D(gray, -1, gabor)

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


# Используйте 6 фильтров Габора с разными углами для извлечения деталей на изображении
def Gabor_process(img, K_size=5, Sigma=2, Gamma=1.75, Lambda=4.25, angles=[0,90]):
    # get shape
    H, W = img.shape
    print(img.shape)

    # gray scale
    if (len(img.shape)>2):
        gray = BGR2GRAY(img).astype(np.float32)
    else:
        gray=img

    out = np.zeros([H, W], dtype=np.float32)

    # each angle
    for angle in angles:
        # gabor filtering
        # _out = Gabor_filtering(gray, K_size=11, Sigma=1.5, Gamma=1.2, Lambda=3, angle = angle)
        # _out = Gabor_filtering(gray, K_size=5, Sigma=1.5, Gamma=1.2, Lambda=3, angle = angle)
        # _out = Gabor_filtering(gray, K_size=5, Sigma=7, Gamma=1.2, Lambda=3, angle = angle)
        # _out = Gabor_filtering(gray, K_size=55, Sigma=7, Gamma=1.5, Lambda=5, angle = angle)
        _out = Gabor_filtering(gray, K_size=K_size, Sigma=Sigma, Gamma=Gamma, Lambda=Lambda, angle = angle)


        # add gabor filtered image
        out += _out

    # scale normalization
    out = out / out.max() * 255
    out = out.astype(np.uint8)

    return out
