import pandas as pd
import os
import glob

# get list of all csv files in current directory
csv_files = glob.glob('dataframe_for_*.csv')

for csv_file in csv_files:
    # extract letter from file name
    letter = os.path.splitext(csv_file)[0][-1]

    # read csv file into a dataframe
    df = pd.read_csv(csv_file)

    # create new column 'letter' with all values set to the extracted letter
    df.insert(1, 'letter', letter)

    # overwrite the existing csv file with updated dataframe
    df.to_csv(csv_file, index=False)
