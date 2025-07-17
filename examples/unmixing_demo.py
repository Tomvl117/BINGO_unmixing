from src.bingo_unmixing.bingo_nmf import BINGONMF
from src.bingo_unmixing.bingo_nmf_gpu import BINGONMF_GPU
import tifffile
import numpy as np


img = tifffile.imread(r"Z:\Rheenen\tvl_jr\normal Merged crop-2.tif")
max_value = np.iinfo(img.dtype).max

model = BINGONMF(
    n_components=img.shape[0],
    alpha=1e-1,
    spH=0.3,
    step_size_h=1e-3,
    max_iter=300,
)

model_gpu = BINGONMF_GPU(
    n_components=img.shape[0],
    alpha=1e-1,
    spH=0.3,
    step_size_h=1e-3,
    max_iter=300,
)

if img.ndim < 4:
    C = img.shape[0]
    X = img.reshape(C, -1).T
# else: 3D application

X = X.astype(float)/max_value

import time

time_start = time.time()
W_cpu = model.fit_transform(X)
print("CPU finished in ", time.time() - time_start)

time_start = time.time()
W_gpu = model_gpu.fit_transform(X)
print("GPU finished in ", time.time() - time_start)

# Output CPU calculated
output = W_cpu.T.reshape(img.shape)
output_img = (output*max_value).astype(img.dtype)

tifffile.imwrite(r"Z:\Rheenen\tvl_jr\unmixed_cpu.tif", output_img)

# Output GPU calculated
output = W_gpu.T.reshape(img.shape)
output_img = (output*max_value).astype(img.dtype)

tifffile.imwrite(r"Z:\Rheenen\tvl_jr\unmixed_gpu.tif", output_img)
