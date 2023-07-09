# American Sign Language Final Project Report MS3

American Sign Language Final Project README

## Project Description
This project aims to develop a deep learning model for American Sign Language (ASL) recognition.
The goal of our project was to develop a robust model for recognizing American Sign Language (ASL) letters, numbers, and symbols from video input and generating a translated character sequence.   
To accomplish this, we participated in the ASFLR Kaggle competition and advanced the state of the art.   
For our approach, we leveraged the dataset provided by the competition, which consisted of 61,955 sequences extracted from various videos.   
We implemented two models for ASL recognition: a baseline LSTM model and a more advanced transformer model.  

## Folder Structure
The repository is organized into the following folders:

- `data`
- `models`:
    * LSTM_model 
    * LSTM_skip
    * Transformer_model
    * Transformer_skip
- `notebooks`:
    * load_dataset.ipynb - a script to load the dataset to Goggle Collaboratory /Kaggle
    * EDA.ipynb - an EDA notebook of the dataset
    * pltDot.ipynb - a notebook that adds the option to transform landmark data to pictures of dots or lines to use with Computer Vision models
- `reports`:
    * Milestone2 - a report on the progress of the project
    * Milestone3 - a report on the progress of the project

## Data
Contains random videos with ASL sequences

## Models
Contains Jupyter notebooks used for data preprocessing, model training, and evaluation.

## Notebooks
The `notebooks` folder contains Jupyter notebooks used throughout the project. These notebooks cover various tasks, including data extraction and preprocessing, model training, and evaluation. They provide a step-by-step guide to reproduce the experiments and understand the project workflow:
 * load_dataset.ipynb - a script to load the dataset to Goggle Collaboratory /Kaggle
 * EDA.ipynb - an EDA notebook of the dataset
 * pltDot.ipynb - a notebook that adds the option to transform landmark data to pictures of dots or lines to use with Computer Vision models


## Reports
Contains project reports we wrote throughout project milestones.

Please refer to the individual folders for more details on their contents and how to use them for the project.