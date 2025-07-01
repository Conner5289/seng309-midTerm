import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
import argparse


# Some debugging stuff
cli_parser = argparse.ArgumentParser()
cli_parser.add_argument("--debug", action="store_true")
args = cli_parser.parse_args()

# Download with kaggle_api instead of downloading the file, doenst work now the owner made the data private
# path = kagglehub.dataset_download(
#     "rakeshkapilavai/extrovert-vs-introvert-behavior-data",
#     path="personality_datasert.csv",
# )
# See where the file is downloaded

# Reads the csv file of data that trains our model:
raw_data = pd.read_csv("personality_datasert.csv")

# delete any rows with empty data before any preprocessing
raw_data.dropna(inplace=True)
# Makes encoder, encodes the data from a dict then kicks the dict to a new data frame with everything
label_encoder = preprocessing.LabelEncoder()
encoded_data_dict = {
    "Time_spent_Alone": raw_data["Time_spent_Alone"],
    "Stage_fear": label_encoder.fit_transform(raw_data["Stage_fear"]),
    "Social_event_attendance": raw_data["Social_event_attendance"],
    "Going_outside": raw_data["Going_outside"],
    "Drained_after_socializing": label_encoder.fit_transform(
        raw_data["Drained_after_socializing"]
    ),
    "Friends_circle_size": raw_data["Friends_circle_size"],
    "Post_frequency": raw_data["Friends_circle_size"],
    "Personality": label_encoder.fit_transform(raw_data["Personality"]),
}
encoded_data = pd.DataFrame(encoded_data_dict)
# Makes a csv file with the encoded data
encoded_data.to_csv("encoded_Data.csv", index=False)

# Feature Scaling, scaled every feature that wasn't label encoded:
# Create a Normalized object from SkLearn:
scaler = Normalizer()

features_to_normalize = [
    "Time_spent_Alone",
    "Social_event_attendance",
    "Going_outside",
    "Friends_circle_size",
    "Post_frequency",
]

norma_array = scaler.fit_transform(encoded_data[features_to_normalize])

norma_dataFrame = pd.DataFrame(norma_array, columns=features_to_normalize)
norma_dataFrame.reset_index(drop=True, inplace=True)

non_norma_columns = encoded_data[
    ["Stage_fear", "Drained_after_socializing", "Personality"]
].reset_index(drop=True)

final_dataframe = pd.concat([norma_dataFrame, non_norma_columns], axis=1)
final_dataframe.to_csv("scaled_data.csv", index=False)

print("Raw data\n", raw_data.head())
print("Encoded_data\n", encoded_data.head())
print("Encoded and normalize data\n", norma_dataFrame.head())


# This shows entire updated/encoded table to ensure it's done correctly
if args.debug:
    print("This is the raw data fromt he csv\n", raw_data)
    print("This is the encoded data\n", encoded_data)
