from transformers import pipeline

# Load the model
instructor = pipeline('zero-shot-classification')

# Define the input
# TODO: Coming from dataset
input = ""

# Define the candidate labels
candidate_labels = ["aggressive", "slow", "normal"]


# Get the prediction
# TODO: THE MODEL NEEDS TO BE FINE-TUNED FIRST



