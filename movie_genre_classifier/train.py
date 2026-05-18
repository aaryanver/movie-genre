import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

def train_model():
    # Load dataset
    try:
        df = pd.read_csv('data.csv')
    except FileNotFoundError:
        print("Error: data.csv not found. Please ensure it exists in the same directory.")
        return

    # Separate features and labels
    X = df[['Duration', 'Rating', 'Votes']]
    y = df['Genre']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the model (K-Nearest Neighbors, k=3)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_scaled, y)

    # Save the model and scaler
    joblib.dump(knn, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    print("Model and scaler successfully saved as 'model.pkl' and 'scaler.pkl'")

if __name__ == '__main__':
    train_model()
