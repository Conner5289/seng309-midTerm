import data_preprocessing as data
import models
import pandas as pd

data_list = data.data_preprocessing()
# gets the processed data, would be nice to make them each in function calls
encoded_dataframe = data_list[0]
norma_dataframe = data_list[1]

encode_label_list = data_list[2]
# getting each label out of the list, need to find a better way
stage_fear_encoder = encode_label_list[0]
draiand_event_encoder = encode_label_list[1]
personality_encoder = encode_label_list[2]

# getting the scaler out
norma_scaler = data_list[3]

print(
    "Welcome to are extrovert or introvert predictor, we have some question for you the we will prodict what you are"
)
# user questions
user_time_alone = input("How many hours do you spend alone each day?: ")
user_stage_fear = input("Do you have stage fright Yes/No: ").title()
user_socail_event = input("How many social event do you go to each week?: ")
user_time_outside = input("How many hours do you spend outside each day?: ")
user_driand_event = input("Do you have feel draind after an event? Yes/No: ").title()
user_friend_num = input("How many friends do you have?: ")
user_post_num = input("Do how many post do you make each day on social media?: ")

# makes a new data dict to make a dataframe and transforms the data into labels
user_data_dict = {
    "Time_spent_Alone": float(user_time_alone),
    "Stage_fear": stage_fear_encoder.transform([user_stage_fear])[0],
    "Social_event_attendance": int(user_socail_event),
    "Going_outside": float(user_time_outside),
    "Drained_after_socializing": draiand_event_encoder.transform([user_driand_event])[
        0
    ],
    "Friends_circle_size": int(user_friend_num),
    "Post_frequency": int(user_post_num),
}
user_dataframe_encoded = pd.DataFrame([user_data_dict])

features_to_normalize = [
    "Time_spent_Alone",
    "Social_event_attendance",
    "Going_outside",
    "Friends_circle_size",
    "Post_frequency",
]

# transforms user input to match the scaled training data
norma_array = norma_scaler.transform(user_dataframe_encoded[features_to_normalize])
norma_dataFrame = pd.DataFrame(norma_array, columns=features_to_normalize)
norma_dataFrame.reset_index(drop=True, inplace=True)

# Only non-normalized categorical columns
non_norma_columns = user_dataframe_encoded[
    ["Stage_fear", "Drained_after_socializing"]
].reset_index(drop=True)

# puts the norma and non norma data back to start order
full_norma_dataframe = pd.concat([norma_dataFrame, non_norma_columns], axis=1)
og_order = [
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Post_frequency",
]
full_norma_dataframe = full_norma_dataframe[og_order]

bad_input = True
while bad_input:
    user_input = int(
        input(
            "Would you like to try our linear regression model(1) or our decision tree model(2)?: "
        )
    )
    # LR
    if user_input == 1:
        bad_input = False
        print("You have choosen to use our linear regression model")
        # makes model
        lr_model = models.linear_model(full_norma_dataframe)
        # makes prediction
        lr_model_prediction = lr_model.predict(full_norma_dataframe)
        # prints anwser, lr_model_prediction is a float from 0 to 1
        if lr_model_prediction < 0.5:
            print("You are an extrovert")
        else:
            print("You are an introvert")

    # DT
    elif user_input == 2:
        bad_input = False
        print("You have choosen to use our decision tree model")
        # makes model
        dt_model = models.decision_tree(encoded_dataframe)
        # makes prediciton
        dt_model_prediction = dt_model.predict(user_dataframe_encoded)
        # Unencodeds the prediction
        predicted = personality_encoder.inverse_transform(dt_model_prediction)
        print("You are an", predicted[0])

    else:
        print("That was not a choice try agian")
