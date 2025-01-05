import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

file_path = "D:\\Perkuliahan\\S5\\Pengolahan Citra Digital\\s12\\praktikums12\\tiger.jpg"
image = imageio.imread(file_path)

if len(image.shape) == 3:
    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]) 

h, w = image.shape
h_blocks = h // 8
w_blocks = w // 8

compressed_data = np.zeros_like(image, dtype=np.float32)

quantization_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
])

for i in range(h_blocks):
    for j in range(w_blocks):
        block = image[i*8:(i+1)*8, j*8:(j+1)*8]
        
        dct_block = dct2(block)
        
        quantized_block = np.round(dct_block / quantization_table)
        
        reconstructed_block = idct2(quantized_block * quantization_table)
        
        compressed_data[i*8:(i+1)*8, j*8:(j+1)*8] = reconstructed_block


compressed_image = np.clip(compressed_data, 0, 255).astype(np.uint8)

plt.imshow(compressed_image, cmap="gray")
plt.axis("off")
plt.show()

output_path = "D:\\Perkuliahan\\S5\\Pengolahan Citra Digital\\s12\\praktikums12\\compressed_image.jpg"
imageio.imwrite(output_path, compressed_image)
print(f"Gambar hasil kompresi telah disimpan dalam format JPEG di: {output_path}")
