import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

# Set random seed
np.random.seed(42)

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

# means = np.load("MEANS.npy")
# stds = np.load("STDS.npy")
X_train = np.load("X_train.npy")
X_val = np.load("X_val.npy")
y_train = np.load("y_train.npy")
y_val = np.load("y_val.npy")

print(stds.shape, means.shape)
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)


def plot_hand(landmarks):
    length_landmarks = len(landmarks)
    x_coords = landmarks[0:int(length_landmarks/2)]
    y_coords = landmarks[int(length_landmarks/2):length_landmarks]
    
    plt.figure(figsize=(5, 5))
    plt.scatter(x_coords, y_coords)
    
    connections = [(0, 1), (1, 2), (2, 3), (3, 4),  # thumb
                   (0, 5), (5, 6), (6, 7), (7, 8),  # index
                   (0, 9), (9, 10), (10, 11), (11, 12),  # middle
                   (0, 13), (13, 14), (14, 15), (15, 16),  # ring
                   (0, 17), (17, 18), (18, 19), (19, 20)]  # pinky
    
    for connection in connections:
        plt.plot([x_coords[connection[0]], x_coords[connection[1]]],
                 [y_coords[connection[0]], y_coords[connection[1]]], 'r')
    
    plt.gca().invert_yaxis()
    plt.show()


land = X_val[6, 60, 0:42]
plot_hand(land)


land = X_val[6, 55, 0:42]
print(len(land))
len1 = land[0:int(len(land)/2)]
len2 = land[int(len(land)/2):len(land)]
print(len(len1))
print(len(len2))
print(len1)



def plot_hand_point(landmarks):
    length_landmarks = len(landmarks)
    x_coords = landmarks[0:int(length_landmarks/2)]
    y_coords = landmarks[int(length_landmarks/2):length_landmarks]
    
    plt.figure(figsize=(5, 5))
    plt.scatter(x_coords, y_coords)
    plt.scatter(x_coords, y_coords, color='black')
    plt.gca().invert_yaxis()
    plt.show()


land = X_val[6, 55, 0:42]
print(len(land))
len1 = land[0:int(len(land)/2)]
len2 = land[int(len(land)/2):len(land)]
print(len(len1))
print(len(len2))
print(len1)


def save_hand_point(landmarks, name_file="hand_plot.png", size=300):
    length_landmarks = len(landmarks)
    x_coords = landmarks[0:int(length_landmarks/2)]
    y_coords = landmarks[int(length_landmarks/2):length_landmarks]
    dpi = 80
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    plt.scatter(x_coords, y_coords, color='black')
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig(name_file, dpi=dpi, pad_inches=0)
    plt.close()


land = X_val[6, 55, 0:42]
save_hand_point(land, "hand_plot.png", 300)


def save_plot_hand(landmarks, name_file="hand_plot.png", size=300):
    length_landmarks = len(landmarks)
    x_coords = landmarks[0:int(length_landmarks/2)]
    y_coords = landmarks[int(length_landmarks/2):length_landmarks]
    dpi = 80
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    plt.scatter(x_coords, y_coords, color='black')
    
    connections = [(0, 1), (1, 2), (2, 3), (3, 4),  # thumb
                   (0, 5), (5, 6), (6, 7), (7, 8),  # index
                   (0, 9), (9, 10), (10, 11), (11, 12),  # middle
                   (0, 13), (13, 14), (14, 15), (15, 16),  # ring
                   (0, 17), (17, 18), (18, 19), (19, 20)]  # pinky
    
    for connection in connections:
        plt.plot([x_coords[connection[0]], x_coords[connection[1]]],
                 [y_coords[connection[0]], y_coords[connection[1]]], 'r')
    
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig(name_file, dpi=dpi, pad_inches=0)
    plt.close()


land = X_val[6, 58, 0:42]
save_plot_hand(land, "hand_plot.png", 300)


def save_lip_point(landmarks, name_file="hand_plot.png", size=300):
    length_landmarks = len(landmarks)
    x_coords = landmarks[0:int(length_landmarks/2)]
    y_coords = landmarks[int(length_landmarks/2):length_landmarks]
    dpi = 80
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    plt.scatter(x_coords, y_coords, color='black')
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig(name_file, dpi=dpi, pad_inches=0)
    plt.close()


len(X_val[6, 58, :])


len(X_val[6, 58, 0:42])


len(X_val[6, 58, 42:84])


len(X_val[6, 58, 84:len(X_val[6, 58, :])])


land = X_val[3, 12, 84:len(X_val[6, 58, :])]
save_lip_point(land, "lips_plot.png", 300)


def plot_lips(landmarks, name_file="lips_plot.png", size=300):
    length_landmarks = len(landmarks)
    x_coords = landmarks[0:int(length_landmarks/2)]
    y_coords = landmarks[int(length_landmarks/2):length_landmarks]
    dpi = 80
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    
    connections = [(7, 17), (17, 15), (15, 19), (19, 12), (12, 3), (3, 30), (30, 37), (37, 33), (33, 35), (35, 25),  # low_low_lip
                   (7, 20), (20, 6), (6, 5), (5, 4), (4, 0), (0, 22), (22, 23), (23, 24), (24, 38), (38, 25),  # up_up_lip
                   (7, 8), (8, 21), (21, 9), (9, 10), (10, 11), (11, 1), (1, 29), (29, 28), (28, 27), (27, 39), (39, 26), (26, 25),  # low_up_lip
                   (7, 8), (8, 16), (16, 14), (14, 18), (18, 13), (13, 2), (2, 31), (31, 36), (36, 32), (32, 34), (34, 26), (26, 25)]  # up_lower_lip
    
    for connection in connections:
        plt.plot([x_coords[connection[0]], x_coords[connection[1]]],
                 [y_coords[connection[0]], y_coords[connection[1]]], 'r')
    
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig(name_file, dpi=dpi, pad_inches=0)
    plt.close()


land = X_val[6, 58, 84:len(X_val[6, 58, :])]
plot_lips(land, "lips_plot.png", 300)


def plot_hand_point_debug(landmarks):
    print(len(landmarks))
    length_landmarks = len(landmarks)
    x_coords = landmarks[0:int(length_landmarks/2)]
    y_coords = landmarks[int(length_landmarks/2):length_landmarks]
    
    plt.figure(figsize=(5, 5))
    plt.scatter(x_coords, y_coords, color='black')
    
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        plt.annotate(i, (x, y), fontsize=12, ha='right')
    
    plt.gca().invert_yaxis()
    plt.show()


land = X_val[11, 11, 84:len(X_val[6, 58, :])]
plot_hand_point_debug(land)
