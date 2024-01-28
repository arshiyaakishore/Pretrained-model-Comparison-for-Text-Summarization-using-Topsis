import pandas as pd
import numpy as np

def load_data(file_path):
    # Add input validation here if needed
    return pd.read_csv(file_path)

def normalize_matrix(data):
    # Extract relevant columns
    rouge_scores = data['Rouge_Scores'].values
    length_of_summary = data['Length_of_Summary'].values
    training_time = data['Training_Time'].values

    # Normalize the matrix
    normalized_matrix = np.column_stack([
        rouge_scores / np.max(rouge_scores),
        1 - (length_of_summary / np.max(length_of_summary)),
        1 - (training_time / np.max(training_time))
    ])
    
    return normalized_matrix

def calculate_topsis_scores(normalized_matrix, weights):
    # Calculate the weighted normalized decision matrix
    weighted_normalized_matrix = normalized_matrix * weights

    # Ideal and Negative Ideal solutions
    ideal_solution = np.max(weighted_normalized_matrix, axis=0)
    negative_ideal_solution = np.min(weighted_normalized_matrix, axis=0)

    # Calculate the separation measures
    distance_to_ideal = np.sqrt(np.sum((weighted_normalized_matrix - ideal_solution)**2, axis=1))
    distance_to_negative_ideal = np.sqrt(np.sum((weighted_normalized_matrix - negative_ideal_solution)**2, axis=1))

    # Calculate the TOPSIS scores
    topsis_scores = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)

    return topsis_scores

def rank_models(data, topsis_scores):
    # Rank the models based on TOPSIS scores
    data['TOPSIS_Score'] = topsis_scores
    data['Rank'] = data['TOPSIS_Score'].rank(ascending=False)

    return data

def main():
    file_path = 'data.csv'
    data = load_data(file_path)

    # Weights for each parameter
    weights = np.array([0.4, 0.3, 0.3])

    # Calculate TOPSIS scores
    normalized_matrix = normalize_matrix(data)
    topsis_scores = calculate_topsis_scores(normalized_matrix, weights)

    # Rank models
    ranked_data = rank_models(data, topsis_scores)

    # Print or save results
    print("Model Ranking:")
    print(ranked_data[['Model', 'TOPSIS_Score', 'Rank']].sort_values(by='Rank'))

    ranked_data.to_csv('result.csv', index=False)

if _name_ == "_main_":
    main()
