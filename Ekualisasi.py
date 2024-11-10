import  numpy as np
import  matplotlib.pyplot as plt
from    scipy import ndimage
import  imageio

def histogram_equalization(img):

  # Hitung histogram
  hist, bins = np.histogram(img.flatten(), 256, [0, 256])

  # Hitung CDF
  cdf = hist.cumsum()
  cdf_normalized = cdf * hist.max() / cdf.max()

  # Buat lookup table
  cdf_m = np.ma.masked_equal(cdf, 0, copy=True)
  cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
  cdf   = np.ma.filled(cdf_m, 0).astype('uint8')

  # Lakukan pemetaan ulang intensitas
  img_equalized = cdf[img]

  return img_equalized

# Membaca citra
img = imageio.imread('contoh.jpeg')

# Ekualisasi histogram
img_equalized = histogram_equalization(img)

# Menampilkan hasil
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Citra Asli')

plt.subplot(1, 2, 2)
plt.imshow(img_equalized, cmap='gray')
plt.title('Citra Hasil Ekualisasi')

plt.show()