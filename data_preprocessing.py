from uuid import main
from numpy._core.multiarray import scalar
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
import argparse


def data_preprocessing():
    # Some debugging stuff
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--debug", action="store_true")
    args = cli_parser.parse_args()

    # Reads the csv file of data that trains our model:
    raw_data = pd.read_csv("personality_datasert.csv")

    # delete any rows with empty data before any preprocessing
    raw_data.dropna(inplace=True)
    # Makes encoder, encodes the data from a dict then kicks the dict to a new data frame with everything
    label_encoder_stage_fear = preprocessing.LabelEncoder()
    label_encoder_drained = preprocessing.LabelEncoder()
    label_encoder_personality = preprocessing.LabelEncoder()
    encoded_data_dict = {
        "Time_spent_Alone": raw_data["Time_spent_Alone"],
        "Stage_fear": label_encoder_stage_fear.fit_transform(raw_data["Stage_fear"]),
        "Social_event_attendance": raw_data["Social_event_attendance"],
        "Going_outside": raw_data["Going_outside"],
        "Drained_after_socializing": label_encoder_drained.fit_transform(
            raw_data["Drained_after_socializing"]
        ),
        "Friends_circle_size": raw_data["Friends_circle_size"],
        "Post_frequency": raw_data["Friends_circle_size"],
        "Personality": label_encoder_personality.fit_transform(raw_data["Personality"]),
    }
    encoded_dataframe = pd.DataFrame(encoded_data_dict)
    encoder_list = [
        label_encoder_stage_fear,
        label_encoder_drained,
        label_encoder_personality,
    ]
    # Makes a csv file with the encoded data
    # encoded_dataframe.to_csv("encoded_Data.csv", index=False)

    # Feature Scaling, scaled every feature that wasn't label encoded:
    # Create a Normalized object from SkLearn:
    data_scaler = Normalizer()

    features_to_normalize = [
        "Time_spent_Alone",
        "Social_event_attendance",
        "Going_outside",
        "Friends_circle_size",
        "Post_frequency",
    ]

    norma_array = data_scaler.fit_transform(encoded_dataframe[features_to_normalize])

    norma_dataFrame = pd.DataFrame(norma_array, columns=features_to_normalize)
    norma_dataFrame.reset_index(drop=True, inplace=True)

    non_norma_columns = encoded_dataframe[
        ["Stage_fear", "Drained_after_socializing", "Personality"]
    ].reset_index(drop=True)

    final_dataframe = pd.concat([norma_dataFrame, non_norma_columns], axis=1)
    # final_dataframe.to_csv("scaled_data.csv", index=False)

    # This shows entire updated/encoded table to ensure it's done correctly
    if args.debug:
        print("This is the raw data fromt he csv\n", raw_data)
        print("This is the encoded data\n", encoded_dataframe)
        print("Raw data\n", raw_data.head())
        print("Encoded_data\n", encoded_dataframe.head())
        print("Encoded and normalize data\n", norma_dataFrame.head())

    return [encoded_dataframe, final_dataframe, data_scaler, encoder_list]


if __name__ == "__main__":
    data_preprocessing()
