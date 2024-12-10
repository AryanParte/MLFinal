import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CarPriceAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.models = {}
        
    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        print(f"\nDataset Shape: {self.data.shape}")
        print(self.data.head())
        print(self.data.info())

    def data_visualization(self):
        # Create a figure for price distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(self.data['Price'], bins=50, kde=True)
        plt.title('Price Distribution')
        plt.xlabel('Price ($)')
        plt.ylabel('Count')
        plt.show()

        # Correlation heatmap of numerical features
        numerical_features = ['Price', 'Levy', 'Prod. year', 'Engine volume', 'Mileage', 'Cylinders', 'Airbags', 'Age', 'Power_Index']
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.data[numerical_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.show()

        # Box plots for categorical variables vs price
        categorical_vars = ['Manufacturer', 'Category', 'Fuel type']
        plt.figure(figsize=(15, 5))
        for i, var in enumerate(categorical_vars, 1):
            plt.subplot(1, 3, i)
            sns.boxplot(x=var, y='Price', data=self.data)
            plt.xticks(rotation=45)
            plt.title(f'Price Distribution by {var}')
        plt.tight_layout()
        plt.show()
    
    def handle_price_outliers(self):
        Q1 = self.data['Price'].quantile(0.25)
        Q3 = self.data['Price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        print(f"\nBefore outlier removal: {len(self.data)} records")
        print(f"Price range: ${self.data['Price'].min():,.2f} - ${self.data['Price'].max():,.2f}")
        
        self.data = self.data[
            (self.data['Price'] >= lower_bound) & 
            (self.data['Price'] <= upper_bound)
        ].copy()
        
        print(f"\nAfter outlier removal: {len(self.data)} records")
        print(f"Price range: ${self.data['Price'].min():,.2f} - ${self.data['Price'].max():,.2f}")

    def preprocess_data(self):
        try:
            # Handle missing values
            self.data['Levy'] = pd.to_numeric(self.data['Levy'].replace('-', '0'), errors='coerce')
            self.data['Levy'] = self.data['Levy'].fillna(self.data['Levy'].median())
            
            # Clean numerical features
            self.data['Mileage'] = self.data['Mileage'].str.replace(' km', '').str.replace(',', '').astype(float)
            self.data['Engine volume'] = self.data['Engine volume'].str.replace(' Turbo', '').astype(float)
            
            # Handle outliers
            self.handle_price_outliers()
            
            # Feature engineering
            self.data['Age'] = 2024 - self.data['Prod. year']
            self.data['Power_Index'] = self.data['Engine volume'] * self.data['Cylinders']
            
            numerical_features = [
                'Levy', 'Prod. year', 'Engine volume', 'Mileage',
                'Cylinders', 'Airbags', 'Age', 'Power_Index'
            ]
            categorical_features = [
                'Manufacturer', 'Category', 'Leather interior',
                'Fuel type', 'Gear box type', 'Drive wheels'
            ]
            
            # Create features
            X_categorical = pd.get_dummies(self.data[categorical_features])
            X = pd.concat([self.data[numerical_features], X_categorical], axis=1)
            y = self.data['Price']
            
            # First split: training and temporary test set (80-20)
            X_temp, self.X_test, y_temp, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Second split: training and validation set (75-25 of the remaining 80%)
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_temp, y_temp, test_size=0.25, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            self.X_train_scaled = scaler.fit_transform(self.X_train)
            self.X_val_scaled = scaler.transform(self.X_val)
            self.X_test_scaled = scaler.transform(self.X_test)
            
            print(f"\nTraining set: {self.X_train.shape}")
            print(f"Validation set: {self.X_val.shape}")
            print(f"Test set: {self.X_test.shape}")
            
        except Exception as e:
            print(f"Preprocessing error: {str(e)}")
            raise

    def train_models(self):
        try:
            # KNN with validation-based tuning
            knn_params = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
            
            knn = KNeighborsRegressor()
            knn_grid = GridSearchCV(knn, knn_params, cv=5, scoring='r2', n_jobs=-1)
            knn_grid.fit(self.X_train_scaled, self.y_train)
            self.models['KNN'] = knn_grid.best_estimator_
            
            # Ridge with validation-based tuning
            ridge_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
            ridge = Ridge()
            ridge_grid = GridSearchCV(ridge, ridge_params, cv=5, scoring='r2', n_jobs=-1)
            ridge_grid.fit(self.X_train_scaled, self.y_train)
            self.models['Ridge'] = ridge_grid.best_estimator_
            
            print("\nBest parameters:")
            print("KNN:", knn_grid.best_params_)
            print("Ridge:", ridge_grid.best_params_)
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise

    def evaluate_models(self):
        print("\nModel Evaluation")
        
        for name, model in self.models.items():
            # Validation set performance
            val_pred = model.predict(self.X_val_scaled)
            val_mae = mean_absolute_error(self.y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))
            val_r2 = r2_score(self.y_val, val_pred)
            
            # Test set performance
            test_pred = model.predict(self.X_test_scaled)
            test_mae = mean_absolute_error(self.y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
            test_r2 = r2_score(self.y_test, test_pred)
            
            print(f"\n{name} Results:")
            print("Validation Set:")
            print(f"MAE: ${val_mae:,.2f} ({val_mae / self.y_val.mean() * 100:.2f}%)")
            print(f"RMSE: ${val_rmse:,.2f} ({val_rmse / self.y_val.mean() * 100:.2f}%)")
            print(f"R² Score: {val_r2:.4f}")
            
            print("\nTest Set:")
            print(f"MAE: ${test_mae:,.2f} ({test_mae / self.y_test.mean() * 100:.2f}%)")
            print(f"RMSE: ${test_rmse:,.2f} ({test_rmse / self.y_test.mean() * 100:.2f}%)")
            print(f"R² Score: {test_r2:.4f}")

    def visualize_model_performance(self):
        for name, model in self.models.items():
            # Predictions vs Actual scatter plot
            plt.figure(figsize=(10, 6))
            test_pred = model.predict(self.X_test_scaled)
            plt.scatter(self.y_test, test_pred, alpha=0.5)
            plt.plot([self.y_test.min(), self.y_test.max()], 
                    [self.y_test.min(), self.y_test.max()], 
                    'r--', lw=2)
            plt.xlabel('Actual Price ($)')
            plt.ylabel('Predicted Price ($)')
            plt.title(f'{name}: Actual vs Predicted Prices')
            plt.tight_layout()
            plt.show()

            # Residual plot
            residuals = test_pred - self.y_test
            plt.figure(figsize=(10, 6))
            plt.scatter(test_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Price ($)')
            plt.ylabel('Residuals ($)')
            plt.title(f'{name}: Residual Plot')
            plt.tight_layout()
            plt.show()

            # Error distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(residuals, bins=50, kde=True)
            plt.xlabel('Prediction Error ($)')
            plt.ylabel('Count')
            plt.title(f'{name}: Error Distribution')
            plt.tight_layout()
            plt.show()

    def visualize_data_distributions(self):
        # Price distribution before outlier removal
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['Price'], bins=50, kde=True)
        plt.title('Price Distribution Before Outlier Removal')
        plt.xlabel('Price ($)')
        plt.ylabel('Count')
        plt.show()
        
        # Numerical features distribution
        num_features = ['Price', 'Levy', 'Prod. year', 'Mileage', 'Cylinders', 'Airbags']
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(num_features, 1):
            plt.subplot(2, 3, i)
            sns.histplot(self.data[feature], bins=30, kde=True)
            plt.title(f'{feature} Distribution')
        plt.tight_layout()
        plt.show()
        
        # Categorical features distribution
        cat_features = ['Manufacturer', 'Category', 'Fuel type', 'Drive wheels']
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(cat_features, 1):
            plt.subplot(2, 2, i)
            value_counts = self.data[feature].value_counts()[:10]  # Top 10 categories
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.xticks(rotation=45)
            plt.title(f'Top 10 {feature} Distribution')
        plt.tight_layout()
        plt.show()

def main():
    try:
        analysis = CarPriceAnalysis('test.csv')
        analysis.load_data()
        analysis.visualize_data_distributions()
        analysis.preprocess_data()
        analysis.train_models()
        analysis.evaluate_models()
        analysis.visualize_model_performance()
    except Exception as e:
        print(f"Execution error: {str(e)}")

if __name__ == "__main__":
    main()