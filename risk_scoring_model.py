from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import pandas as pd
from typing import Dict, Optional
import os

class RiskScoringModel:
    # Handles training the anomaly detection model and assigning risk scores to transactions.
    # Implements Isolation Forest for anomaly detection.
    def __init__(self, model_type: str = "isolation_forest", model_params: Optional[Dict] = None):
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None
        self.preprocessor = None
        self.numeric_features = ['amount']
        self.categorical_features = ['department', 'expense_type', 'vendor', 'payment_method', 'employee']
        self.date_features = ['day_of_week', 'hour_of_day', 'is_weekend', 'is_end_of_month']
        
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # Preprocess transaction data for model training/prediction.
        # Dataframe of transaction data 
        # Returns reprocessed Dataframe
        df = data.copy()
        # Convert date to datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        # Extract date features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['hour_of_day'] = df['date'].dt.hour
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df['is_end_of_month'] = df['date'].dt.is_month_end.astype(int)
        return df
    
    def _create_preprocessor(self) -> ColumnTransformer:
        # Create a preprocessor for the features
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        preprocessor = ColumnTransformer(
            transformers=[('num', numeric_transformer, self.numeric_features + self.date_features),('cat', categorical_transformer, self.categorical_features)])
        return preprocessor
    
    def train(self, data: pd.DataFrame) -> None:
        #Train the risk scoring model on transaction data.
        df = self.preprocess_data(data)
        # Create and fit preprocessor
        self.preprocessor = self._create_preprocessor()
        # Initialize model based on model_type
        if self.model_type == "isolation_forest":
            params = {
                'n_estimators': self.model_params.get('n_estimators', 100),
                'max_samples': self.model_params.get('max_samples', 'auto'),
                'contamination': self.model_params.get('contamination', 0.05),
                'random_state': self.model_params.get('random_state', 42),
                'n_jobs': self.model_params.get('n_jobs', -1)
            }
            self.model = IsolationForest(**params)
        elif self.model_type in ["lstm", "gru"]:
            # Would implement LSTM or GRU models here using TensorFlow/Keras
            raise NotImplementedError(f"{self.model_type} model is not implemented yet")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        # Prepare features (excluding transaction_id, date, and description)
        features = df[self.numeric_features + self.categorical_features + self.date_features]
        # Fit preprocessor and transform data
        X = self.preprocessor.fit_transform(features)
        # Train model
        self.model.fit(X)
        print(f"Trained {self.model_type} model successfully")
        
    def predict_risk_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        #Predict anomaly scores for transaction data.
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model not trained. Call train() first.")
        df = self.preprocess_data(data)
        # Prepare features
        features = df[self.numeric_features + self.categorical_features + self.date_features]
        # Transform data
        X = self.preprocessor.transform(features)
        # Predict anomaly scores
        if self.model_type == "isolation_forest":
            # Decision scores -> negative = anomaly, positive = normal
            decision_scores = self.model.decision_function(X)
            # Convert to risk scores (0-100)
            # Map from decision function to 0-100 scale
            risk_scores = 100 * (1 - (decision_scores - min(decision_scores))/(max(decision_scores) - min(decision_scores)))
            # Add risk scores to original data
            result_df = data.copy()
            result_df['risk_score'] = risk_scores
            # Add risk categories
            result_df['risk_category'] = pd.cut(risk_scores,bins=[0, 20, 40, 60, 80, 100],labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            return result_df
        elif self.model_type in ["lstm", "gru"]:
            # Would implement LSTM or GRU prediction here
            raise NotImplementedError(f"{self.model_type} prediction not implemented yet")
        return data
    
    def save_model(self, filepath: str) -> None:
        # Save the trained model and preprocessor to a file
        if self.model is None:
            raise ValueError("No trained model to save")
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'model_type': self.model_type,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'date_features': self.date_features
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath: str) -> None:
        # Load a trained model and preprocessor from a file
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.model_type = model_data['model_type']
        self.numeric_features = model_data['numeric_features']
        self.categorical_features = model_data['categorical_features']
        self.date_features = model_data['date_features']
        print(f"Model loaded from {filepath}")