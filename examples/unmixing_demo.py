from src.bingo_unmixing.bingo_nmf import BINGONMF
from src.bingo_unmixing.bingo_nmf_gpu import BINGONMF_GPU
import tifffile


img = tifffile.imread(r"Z:\Rheenen\tvl_jr\normal Merged crop-2.tif")

model = BINGONMF(
    n_components=img.shape[0],
    alpha=1e-1,
    spH=0.5,
    step_size_h=1e-3,
    max_iter=300,
)

model_gpu = BINGONMF_GPU(
    n_components=img.shape[0],
    alpha=1e-1,
    step_size_h=1e-3,
    max_iter=300,
)

import time

time_start = time.time()
W_cpu = model.fit_transform(img)
print("CPU finished in ", time.time() - time_start)

time_start = time.time()
W_gpu = model_gpu.fit_transform(img)
print("GPU finished in ", time.time() - time_start)

# Output CPU calculated
tifffile.imwrite(r"Z:\Rheenen\tvl_jr\unmixed_cpu.tif", W_cpu)

# Output GPU calculated
tifffile.imwrite(r"Z:\Rheenen\tvl_jr\unmixed_gpu.tif", W_gpu)
