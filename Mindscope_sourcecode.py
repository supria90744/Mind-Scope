#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

def generate_random_data(num_samples=200):
    np.random.seed(42)  # For reproducibility

    # Generate random numerical features
    heart_beat_rate = np.random.randint(60, 100, num_samples)
    sleep_hours = np.random.uniform(4, 10, num_samples)
    number_of_steps_walked = np.random.randint(1000, 15000, num_samples)
    mental_crisis = np.random.randint(0, 2, num_samples)  # Binary target variable

    # Generate random categorical features
    anger_level = np.random.randint(1, 6, num_samples)  # Levels from 1 to 5
    anxiety_level = np.random.randint(1, 6, num_samples)
    excitement_level = np.random.randint(1, 6, num_samples)
    current_feeling = np.random.choice(['Happy', 'Sad', 'Neutral'], num_samples)

    # Create DataFrame
    data = pd.DataFrame({
        'Heart_beat_rate': heart_beat_rate,
        'Sleep_hours': sleep_hours,
        'Number_of_steps_walked': number_of_steps_walked,
        'Mental_crisis': mental_crisis,
        'Anger_level': anger_level,
        'Anxiety_level': anxiety_level,
        'Excitement_level': excitement_level,
        'Your_current_feeling': current_feeling
    })

    return data

# Example usage
if __name__ == "__main__":
    random_data = generate_random_data()
    random_data.to_csv('random_mentalhealth.csv', index=False)
    print("Random dataset generated and saved to 'random_mentalhealth.csv'.")


# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def load_and_prepare_data(filepath):
    """Load data and handle missing values."""
    data = pd.read_csv(filepath)
    print("Columns in DataFrame:", data.columns)

    # Define features
    numerical_features = ['Heart_beat_rate', 'Sleep_hours', 'Number_of_steps_walked', 'Mental_crisis']
    categorical_features = ['Anger_level', 'Anxiety_level', 'Excitement_level', 'Your_current_feeling']

    # Impute missing values
    numerical_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    data[numerical_features] = numerical_imputer.fit_transform(data[numerical_features])
    data[categorical_features] = categorical_imputer.fit_transform(data[categorical_features])

    # Remove outliers
    z_scores = np.abs(stats.zscore(data[numerical_features]))
    data = data[(z_scores < 3).all(axis=1)]

    # Encode categorical features
    data = pd.get_dummies(data, columns=categorical_features, drop_first=True)
    return data

def prepare_features_and_target(data):
    """Prepare features and target variable."""
    X = data.drop('Mental_crisis', axis=1)
    y = data['Mental_crisis']
    return X, y

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate models."""
    models = {
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(),
        'Linear Regression': LinearRegression(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(),
        'Neural Network': MLPClassifier(max_iter=1000)
    }

    accuracies = {}
    for name, model in models.items():
        # Special handling for Linear Regression
        if name == 'Linear Regression':
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred = [round(value) for value in y_pred]  # Convert to binary
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies[name] = accuracy

        # Print confusion matrix and accuracy
        print(f'{name} Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
        print(f'{name} Accuracy: {accuracy:.4f}')
    
    return accuracies

def plot_results(accuracies, y):
    """Plot the results."""
    # Count of depression
    depressed_count = sum(y)
    non_depressed_count = len(y) - depressed_count

    # Plot pie chart for depression count
    labels = ['In Depression', 'Not in Depression']
    sizes = [depressed_count, non_depressed_count]
    colors = ['#ff9999', '#66b3ff']
    explode = (0.1, 0)  # explode the 1st slice

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title('Depression Count')
    plt.show()

    # Plot bar chart for accuracy
    plt.figure(figsize=(12, 6))
    plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
    plt.xlabel('Algorithm')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Different Algorithms')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def main():
    filepath = 'random_mentalhealth.csv'
    data = load_and_prepare_data(filepath)
    X, y = prepare_features_and_target(data)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train and evaluate models
    accuracies = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Plot results
    plot_results(accuracies, y)

if __name__ == "__main__":
    main()


# In[ ]:




