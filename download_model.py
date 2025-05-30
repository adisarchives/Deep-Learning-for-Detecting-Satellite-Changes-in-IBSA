import gdown
import os

# Directory to store downloaded files
os.makedirs("model", exist_ok=True)

# File 1: CNN Weights
url_cnn = "https://drive.google.com/uc?id=1LZOycnE96XNNPXhYsrI6rJ5pv_j5Ezq7"
output_cnn = "model/cnn_weights.hdf5"
gdown.download(url_cnn, output_cnn, quiet=False)

# File 2: X.npy
url_x = "https://drive.google.com/uc?id=1t7ifVPiQgOkaLnhtoOImrOwbERpYC2W7"
output_x = "model/X.npy"
gdown.download(url_x, output_x, quiet=False)
