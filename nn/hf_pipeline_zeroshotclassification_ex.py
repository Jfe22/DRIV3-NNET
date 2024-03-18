from transformers import pipeline

# Load the model
instructor = pipeline('zero-shot-classification')

# Define the input
input_text = "This is a text about politics"

# Define the candidate labels
candidate_labels = ["politics", "health", "economy"]

# Get the prediction
print(instructor(input_text, candidate_labels))
