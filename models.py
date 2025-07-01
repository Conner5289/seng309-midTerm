from json import encoder

from pandas.core.common import random_state
from sklearn.base import re
import data_preprocessing as data
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data_frames = data.data_preprocessing()
encoded_data = data_frames[0]
norma_data = data_frames[1]


def norma_encode():
    encoder_list = data_frames[3]
    norma = data_frames[2]
    return [encoder_list, norma]


def linear_model():
    # Start of linear_model
    #
    # Time_spent_Alone,Stage_fear,Social_event_attendance,Going_outside,Drained_after_socializing,Friends_circle_size,Post_frequency,Personality
    feature_data = norma_data[
        [
            "Time_spent_Alone",
            "Stage_fear",
            "Social_event_attendance",
            "Going_outside",
            "Drained_after_socializing",
            "Friends_circle_size",
            "Post_frequency",
        ]
    ]

    target = norma_data["Personality"]

    # ran state is set to make it runt he same everytime
    feature_data_train, feature_data_test, target_train, target_test = train_test_split(
        feature_data, target, test_size=0.2, random_state=52
    )

    lr_model = LinearRegression()
    lr_model.fit(feature_data_train, target_train)

    return lr_model

    # lr_predict = lr_model.predict(feature_data_test)


def decision_tree():
    feature_data = encoded_data[
        [
            "Time_spent_Alone",
            "Stage_fear",
            "Social_event_attendance",
            "Going_outside",
            "Drained_after_socializing",
            "Friends_circle_size",
            "Post_frequency",
        ]
    ]
    target = encoded_data["Personality"]

    feature_data_train, feature_data_test, target_train, target_test = train_test_split(
        feature_data, target, test_size=0.2, random_state=52
    )

    dt_model = DecisionTreeClassifier(random_state=52)
    dt_model.fit(feature_data_train, target_train)

    return dt_model
