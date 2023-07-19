import subprocess

# Install gdown package
subprocess.call(['pip3', 'install', 'gdown'])

import gdown

# Download X_train
gdown.download('https://drive.google.com/uc?id=19JTV9whllh5JaW1BkDpu4FoJU_0dpvjq', 'X_train.npy')

# Download X_val
gdown.download('https://drive.google.com/uc?id=15vKRrt0SzXDPfbyeLuOZ0yiUMTlyFbGm', 'X_val.npy')

# Download y_train
gdown.download('https://drive.google.com/uc?id=1d1SVglkffNSZALCvoyMlgiDMDA2DsVXK', 'y_train.npy')

# Download y_val
gdown.download('https://drive.google.com/uc?id=1PhIZZCgfA3TqFIUjJl-z80Au6LJN84bw', 'y_val.npy')

# Check the downloaded files
import os
files = os.listdir()
print(files)
