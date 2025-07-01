from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def linear_model(norma_data):
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


def decision_tree(encoded_data):
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
