import numpy as np
import pandas as pd
from sklearn import preprocessing

pd.set_option('display.max_columns', None)

# Reads the csv file of data that trains our model:
df = pd.read_csv("C:\\Users\\skkae\\Downloads\\personality_datasert.csv")

# Encoding our data
# Label Encoding "Stage_fear":
# Creates a LabelEncoder object that converts text like yes or no into numbers like 1 and 0
label_encoder = preprocessing.LabelEncoder()

# .fit() tells the encoder to learn the unique values in stage_fear
# .transform() replaces those unique values with numeric code
df['Stage_fear']= label_encoder.fit_transform(df['Stage_fear'])


# Label Encoding "Drained_after_socializing":
label_encoder = preprocessing.LabelEncoder()
df['Drained_after_socializing']= label_encoder.fit_transform(df['Drained_after_socializing'])
df['Drained_after_socializing'].unique()

# Label Encoding "Personality":
label_encoder = preprocessing.LabelEncoder()
df['Personality']= label_encoder.fit_transform(df['Personality'])
df['Personality'].unique()

# This shows entire updated/encoded table to ensure it's done correctly
# Can delete later - it's only for testing
print(df)

