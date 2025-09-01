import numpy as np
from collections import Counter

def get_entropy_of_dataset(data: np.ndarray) -> float:
    """
    Calculate the entropy of the entire dataset using the target variable (last column).
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
    
    Returns:
        float: Entropy value calculated using the formula: 
               Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i
    
    Example:
        data = np.array([[1, 0, 'yes'],
                        [1, 1, 'no'],
                        [0, 0, 'yes']])
        entropy = get_entropy_of_dataset(data)
        # Should return entropy based on target column ['yes', 'no', 'yes']
    """
    # TODO: Implement entropy calculation
    # Hint: Use np.unique() to get unique classes and their counts
    # Hint: Handle the case when probability is 0 to avoid log2(0)
    if data.shape[0] == 0:  # Checking for an empty dataset.
        return 0.0
    target_col = data[:, -1] # Last column
    _, counts = np.unique(target_col, return_counts=True) # Getting the count of all uniqe values in target class, eg: Yes and No counts.
    probabilities = counts / data.shape[0] # data.shape -> Getting the total number of rows.
    probabilities = probabilities[probabilities > 0] # Taking all probabilities greater than 0 to avoid log2(0).
    entropy = 0.0
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def get_avg_info_of_attribute(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the average information (weighted entropy) of a specific attribute.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
        attribute (int): Index of the attribute column to calculate average information for
    
    Returns:
        float: Average information calculated using the formula:
               Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) 
               where S_v is subset of data with attribute value v
    
    Example:
        data = np.array([[1, 0, 'yes'],
                        [1, 1, 'no'],
                        [0, 0, 'yes']])
        avg_info = get_avg_info_of_attribute(data, 0)  # For attribute at index 0
        # Should return weighted average entropy for attribute splits
    """
    # TODO: Implement average information calculation
    # Hint: For each unique value in the attribute column:
    #   1. Create a subset of data with that value
    #   2. Calculate the entropy of that subset
    #   3. Weight it by the proportion of samples with that value
    #   4. Sum all weighted entropies
    if data.shape[0] == 0 or attribute < 0 or attribute >= data.shape[1] - 1:
        return 0.0
    attribute_column = data[:, attribute] # Picking that particular attribute.
    unique_values = np.unique(attribute_column)
    avg_info = 0.0
    for value in unique_values: # Analyse the corresponding subsets for the attribute.
        mask = attribute_column == value
        subset = data[mask] # Contains only the rows that have that attribute.
        weight = subset.shape[0] / data.shape[0]
        if subset.shape[0] > 0: # Calculating the entropy of just that subset.
            subset_entropy = get_entropy_of_dataset(subset)
            avg_info += weight * subset_entropy
    return avg_info

def get_information_gain(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the Information Gain for a specific attribute.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
        attribute (int): Index of the attribute column to calculate information gain for
    
    Returns:
        float: Information gain calculated using the formula:
               Information_Gain = Entropy(S) - Avg_Info(attribute)
               Rounded to 4 decimal places
    
    Example:
        data = np.array([[1, 0, 'yes'],
                        [1, 1, 'no'],
                        [0, 0, 'yes']])
        gain = get_information_gain(data, 0)  # For attribute at index 0
        # Should return the information gain for splitting on attribute 0
    """
    # TODO: Implement information gain calculation
    # Hint: Information Gain = Dataset Entropy - Average Information of Attribute
    # Hint: Use the functions you implemented above
    # Hint: Round the result to 4 decimal places
    if data.shape[0] == 0:
        return 0.0
    ds_entropy = get_entropy_of_dataset(data) # Entropy of the dataset before the split.
    avg_info = get_avg_info_of_attribute(data, attribute) # Entropy after splitting at the attribute.
    info_gain = ds_entropy - avg_info
    return round(info_gain, 4)

def get_selected_attribute(data: np.ndarray) -> tuple:
    """
    Select the best attribute based on highest information gain.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
    
    Returns:
        tuple: A tuple containing:
            - dict: Dictionary mapping attribute indices to their information gains
            - int: Index of the attribute with the highest information gain
    
    Example:
        data = np.array([[1, 0, 2, 'yes'],
                        [1, 1, 1, 'no'],
                        [0, 0, 2, 'yes']])
        result = get_selected_attribute(data)
        # Should return something like: ({0: 0.123, 1: 0.456, 2: 0.789}, 2)
        # where 2 is the index of the attribute with highest gain
    """
    # TODO: Implement attribute selection
    # Hint: Calculate information gain for all attributes (except target variable)
    # Hint: Store gains in a dictionary with attribute index as key
    # Hint: Find the attribute with maximum gain using max() with key parameter
    # Hint: Return tuple (gain_dictionary, selected_attribute_index)
    if np.unique(data[:, -1]).shape[0] == 1: # Node is already pure, no need to split.
        return ({}, -1)
    if data.shape[0] == 0 or data.shape[1] <= 1:
        return ({}, -1)
    attr_count = data.shape[1] - 1 # Calculate the info gain for all attributes for all columns except the target variable.
    gain_dictionary = {i: get_information_gain(data, i) for i in range(attr_count)}
    if not gain_dictionary:
        return ({}, -1)
    max_index = max(gain_dictionary, key=gain_dictionary.get) # Getting the attribute with the highest info gain.
    return (gain_dictionary, max_index) # Returning both the dictionary with the mapping of all attributes with its info gain along with the index of the attribute with the highest info gain.
