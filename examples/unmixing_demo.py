from src.bingo_unmixing.bingo_nmf import BINGONMF
from sklearn.preprocessing import normalize
import tifffile
import numpy as np


img = tifffile.imread(r"Z:\Rheenen\tvl_jr\normal Merged crop-1.tif")
max_value = np.iinfo(img.dtype).max

model = BINGONMF(
    n_components=img.shape[0],
    alpha_h=1e-3,
    step_size_w=1e-3,
    step_size_h=1e-3,
    max_iter=300,
    random_state=0
)

if img.ndim < 4:
    C = img.shape[0]
    X = img.reshape(C, -1).T
# else: 3D application
X = normalize(X)

W = model.fit_transform(X)
H = model.components_

output = W.T.reshape(img.shape)
output_img = output.astype(img.dtype)

tifffile.imwrite(r"Z:\Rheenen\tvl_jr\unmixed.tif", output_img)

# abundance_maps.shape == (10, 100, 100)
# spectra.shape        == (10, 10)
