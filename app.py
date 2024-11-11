import streamlit as st
import mlflow
import os
import pandas as pd

# Load the model from MLflow using the run ID
run_id = '08bb8cea198748938eeefb1e4f2dea61'  # Replace this with your actual run ID
model_name = 'logistic_regression_model'
vectorized_name = 'vectorizer'

model_uri = f"runs:/{run_id}/{model_name}"
vectorizer_uri = f"runs:/{run_id}/{vectorized_name}"

# Load the model
model = mlflow.sklearn.load_model(model_uri)
vectorizer = mlflow.sklearn.load_model(vectorizer_uri)

# Function to log feedback to a CSV file
def log_feedback_to_csv(text, prediction, feedback):
    # Check if the file exists
    if not os.path.exists('feedback_data.csv'):
        # If it doesn't exist, create it with column headers
        with open('feedback_data.csv', 'w') as f:
            f.write('text,prediction,feedback\n')

    # Append the new feedback to the CSV file
    with open('feedback_data.csv', 'a') as f:
        f.write(f"{text},{prediction},{feedback}\n")

# Function to make predictions using the loaded model
def predict(input_text):
    transformed_text = vectorizer.transform([input_text])
    # Reshape the input into 2D array format (1 sample, -1 feature)
    prediction = model.predict(transformed_text)[0]  # Reshape to 2D array
    return prediction

# Streamlit UI
st.title('CYBERBULLYING DETECTION')
st.subheader('This platform will detect if a sentence contain toxic/attack/bullying meaning')

# User input for the sentence to predict
user_input = st.text_area("Please enter a sentence:", "")

if st.button("Predict"):
    if user_input:
        # Make the prediction using the MLflow-loaded model
        prediction = predict(user_input)
        label = "Toxic" if prediction == 1 else "Not Toxic"
        st.success(f"Prediction: {label}")

        # Feedback options for the user
        st.write("Does the prediction make sense?")
        if st.button("Yes"):
            st.success("Thank you for your feedback!")
            log_feedback_to_csv(user_input, prediction, "Yes")
        elif st.button("No"):
            st.write("Sorry about that. We'll improve!")
            log_feedback_to_csv(user_input, prediction, "No")
