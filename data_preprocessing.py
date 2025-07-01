import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
import argparse


# Need call from main, brake encode and norma to there own func and
def data_preprocessing():
    raw_data = pd.read_csv("personality_datasert.csv")

    raw_data.dropna(inplace=True)

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

    final_norma_dataframe = pd.concat([norma_dataFrame, non_norma_columns], axis=1)

    return [encoded_dataframe, final_norma_dataframe, data_scaler, encoder_list]
