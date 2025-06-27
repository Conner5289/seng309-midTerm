import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer

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



# Feature Scaling, scaled every feature that wasn't label encoded:
# Create a Normalized object from SkLearn:
scaler = Normalizer()

# Making a list of only the features we want to scale, not the one's we Label Encoded
features_to_scale = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                     'Friends_circle_size', 'Post_frequency']

# Grab the data from those columns and normalize them
#.fit() learns the structure
#.transform() applies normalization to each row
scaled_data = scaler.fit_transform(df[features_to_scale])

# Create a new Dataframe with names of the columns the same so it looks like the original table
scaled_df = pd.DataFrame(scaled_data, columns=features_to_scale)

#Prints the first 5 rows
print(scaled_df.head())



# This shows entire updated/encoded table to ensure it's done correctly
# Can delete later - it's only for testing
print(df)

