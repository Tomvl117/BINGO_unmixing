from src.bingo_unmixing.bingo_nmf import BINGONMF
import tifffile
import numpy as np


img = tifffile.imread(r"Z:\Rheenen\tvl_jr\normal Merged crop-2.tif")
max_value = np.iinfo(img.dtype).max

model = BINGONMF(
    n_components=img.shape[0],
    alpha=1e-1,
    spH=0.3,
    step_size_w=1e-3,
    step_size_h=1e-3,
    max_iter=300,
    random_state=0
)

if img.ndim < 4:
    C = img.shape[0]
    X = img.reshape(C, -1).T
# else: 3D application

X = X.astype(float)/max_value

W = model.fit_transform(X)
H = model.components_

output = W.T.reshape(img.shape)
output_img = (output*max_value).astype(img.dtype)

tifffile.imwrite(r"Z:\Rheenen\tvl_jr\unmixed-float.tif", output)
tifffile.imwrite(r"Z:\Rheenen\tvl_jr\unmixed.tif", output_img)
