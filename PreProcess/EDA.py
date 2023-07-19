# Import Libraries

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
import tensorflow as tf
import os
import warnings
warnings.filterwarnings(action='ignore')
# Inspect the 'PARQUET' data file
import subprocess
import gdown

# Sign Language EDA

# Our goal is to classify American Sign Language (ASL) alphabet video files with short sentences using lips/hand landmarks.
# Here is an example of one landmarks data..
# <img src="https://developers.google.com/static/mediapipe/images/solutions/hand-landmarks.png" width="750">
# <p><a href="https://developers.google.com/mediapipe/solutions/vision/hand_landmarker">reference</a></p>


# Loading Dataset

df = pd.read_csv("train.csv")

# Data Description

print(df.head())

# The phrases in the training set contain random websites/addresses/phone numbers.

print(f"Total number of files: {df.shape[0]}")
print(f"Total number of participants in the dataset: {df.participant_id.nunique()}")
print(f"Total number of unique phrases: {df.phrase.nunique()}")

# Data Dictionary

# `path` - The path to the landmark file.
# `file_id` - A unique identifier for the data file.
# `participant_id` - A unique identifier for the data contributor.
# `sequence_id` - A unique identifier for the landmark sequence. Each data file may contain many sequences.
# `phrase` - The labels for the landmark sequence. real addresses/phone numbers/urls.

print("Dimensions of the dataset: ", df.shape)
print()
print("Data types of columns: \n")
df.info()

# EDA - Exploratory Data Analysis

# Inspect the 'PATH' Column

print(np.array(list(df["path"].value_counts().to_dict().values())).min())

print(df["path"].describe().to_frame().T)

# The path column is the path to the landmark file (parquet).
# Number of unique paths: 68 - there are 68 unique parquet files in the directory
# Minimum number of repeated path is: 287 - The minimum amount of "sentences" in a parquet file
# Maximum number of repeated path is: 1000  - The maximum amount of "sentences" in a parquet file

# Inspect the 'PARTICIPANT_ID' Column

print(np.array(list(df["participant_id"].value_counts().to_dict().values())).min())
print(np.array(list(df["participant_id"].value_counts().to_dict().values())).max())

# The participant_id statistics indicate a varied distribution of data contributions among participants, with some participants contributing more examples than others.
# Number of Unique Participants: 94
# Minimum Number of Examples For One Participant: 1
# Maximum Number of Examples For One Participant: 1537

# Set the column 'participant_id' to string type as it is an ID
df["participant_id"] = df["participant_id"].astype(str)

# Calculate the counts for each participant_id
counts = df["participant_id"].value_counts()

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(15, 6))

# Plot the histogram
bars = ax.bar(counts.index, counts.values)

# Set the labels and title
ax.set_xlabel("Participant ID")
ax.set_ylabel("Number of submissions")
ax.set_title("Submission Counts by Participant ID")

# Rotate the x-axis labels if needed
plt.xticks(rotation=90, ha='center')
plt.xlim(-1, 94)

# Show the plot
plt.show()

# Inspect the 'SEQUENCE_ID' Column

print(df["sequence_id"].astype(str).describe().to_frame().T)

# A unique identifier for the landmark sequence.
# Each parquet file may contain many sequences.
# Every value is unique for every row.

# Inspect the 'PHRASE' Column

sns.histplot(df.phrase.str.len(), kde=True, binwidth=2)
plt.title('Count of Phrase length in training set')
plt.xlabel('Phrase length')
plt.ylabel('Sample Count')
plt.tight_layout()
plt.show()

# The encoding and the characters used by the dataset

# Read Character to Ordinal Encoding Mapping
with open('character_to_prediction_index.json') as json_file:
    CHAR2ORD = json.load(json_file)

# Character to Ordinal Encoding Mapping
print("pd.Series(CHAR2ORD).to_frame('Ordinal Encoding')")



# Install gdown package
subprocess.call(['pip3', 'install', 'gdown'])

# Download 5414471.parquet
gdown.download('https://drive.google.com/uc?id=1G6h63_DzYHr2Z5A-StsHK3QGWUIYATyI', '5414471.parquet')

sample = pd.read_parquet("5414471.parquet")

print(f"Sample shape = {sample.shape}")

print(f"Sample shape = {sample.columns}")
