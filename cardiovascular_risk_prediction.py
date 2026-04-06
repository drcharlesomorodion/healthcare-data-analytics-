"""
Cardiovascular Disease Risk Prediction Model
============================================
Machine learning model to predict cardiovascular disease risk based on
patient demographics, clinical indicators, and lifestyle factors.

Author: Dr. Charles Osahenrumwen Omorodion
Date: April 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CardiovascularRiskPredictor:
    """
    A class to predict cardiovascular disease risk using multiple ML algorithms.
    """
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self, filepath):
        """Load and validate the cardiovascular dataset."""
        print("Loading dataset...")
        self.data = pd.read_csv(filepath)
        print(f"Dataset loaded: {self.data.shape[0]} records, {self.data.shape[1]} features")
        return self.data
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic statistics
        print("\nDataset Overview:")
        print(self.data.info())
        print("\nDescriptive Statistics:")
        print(self.data.describe())
        
        # Check for missing values
        missing = self.data.isnull().sum()
        if missing.sum() > 0:
            print("\nMissing Values:")
            print(missing[missing > 0])
        else:
            print("\nNo missing values found.")
        
        # Target distribution
        print("\nTarget Distribution:")
        print(self.data['cardio'].value_counts())
        print(f"Cardiovascular Disease Rate: {self.data['cardio'].mean():.2%}")
        
        return self.data
    
    def visualize_data(self):
        """Create visualizations for data understanding."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Age distribution by cardiovascular disease
        sns.histplot(data=self.data, x='age', hue='cardio', bins=30, ax=axes[0,0])
        axes[0,0].set_title('Age Distribution by CVD Status')
        axes[0,0].set_xlabel('Age (days)')
        
        # Blood pressure distribution
        sns.scatterplot(data=self.data, x='ap_hi', y='ap_lo', hue='cardio', 
                       alpha=0.5, ax=axes[0,1])
        axes[0,1].set_title('Blood Pressure Distribution')
        axes[0,1].set_xlabel('Systolic BP')
        axes[0,1].set_ylabel('Diastolic BP')
        
        # Cholesterol levels
        sns.countplot(data=self.data, x='cholesterol', hue='cardio', ax=axes[0,2])
        axes[0,2].set_title('Cholesterol Levels by CVD Status')
        
        # BMI calculation and distribution
        self.data['bmi'] = self.data['weight'] / (self.data['height']/100)**2
        sns.boxplot(data=self.data, x='cardio', y='bmi', ax=axes[1,0])
        axes[1,0].set_title('BMI Distribution by CVD Status')
        
        # Correlation heatmap
        numeric_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'cardio']
        corr_matrix = self.data[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
        axes[1,1].set_title('Feature Correlation Matrix')
        
        # Feature importance placeholder
        axes[1,2].text(0.5, 0.5, 'Feature Importance\n(Generated after model training)',
                      ha='center', va='center', fontsize=12)
        axes[1,2].set_title('Feature Importance')
        
        plt.tight_layout()
        plt.savefig('cardiovascular_eda.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nVisualizations saved as 'cardiovascular_eda.png'")
    
    def preprocess_data(self):
        """Preprocess data for machine learning."""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Feature engineering
        self.data['bmi'] = self.data['weight'] / (self.data['height']/100)**2
        
        # Create age in years
        self.data['age_years'] = self.data['age'] / 365.25
        
        # Blood pressure categories
        self.data['bp_category'] = pd.cut(
            self.data['ap_hi'],
            bins=[0, 120, 140, 160, 300],
            labels=['Normal', 'Elevated', 'Stage1', 'Stage2']
        )
        
        # Select features
        feature_cols = ['age_years', 'gender', 'height', 'weight', 'bmi',
                       'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 
                       'smoke', 'alco', 'active']
        
        X = self.data[feature_cols]
        y = self.data['cardio']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Features: {self.X_train.shape[1]}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_logistic_regression(self):
        """Train and evaluate logistic regression model."""
        print("\n" + "="*50)
        print("LOGISTIC REGRESSION MODEL")
        print("="*50)
        
        # Train model
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(self.X_train_scaled, self.y_train)
        
        # Predictions
        y_pred = lr_model.predict(self.X_test_scaled)
        y_pred_proba = lr_model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Evaluation
        accuracy = accuracy_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Store results
        self.models['Logistic Regression'] = lr_model
        self.results['Logistic Regression'] = {
            'accuracy': accuracy,
            'auc': auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Feature coefficients
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'coefficient': lr_model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print("\nTop Risk Factors (Logistic Regression Coefficients):")
        print(feature_importance.head(10))
        
        return lr_model
    
    def train_random_forest(self):
        """Train and evaluate random forest model."""
        print("\n" + "="*50)
        print("RANDOM FOREST MODEL")
        print("="*50)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Train with best parameters
        rf_model = grid_search.best_estimator_
        
        # Predictions
        y_pred = rf_model.predict(self.X_test)
        y_pred_proba = rf_model.predict_proba(self.X_test)[:, 1]
        
        # Evaluation
        accuracy = accuracy_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Store results
        self.models['Random Forest'] = rf_model
        self.results['Random Forest'] = {
            'accuracy': accuracy,
            'auc': auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance (Random Forest):")
        print(feature_importance.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), y='feature', x='importance')
        plt.title('Top 10 Feature Importance (Random Forest)')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return rf_model
    
    def compare_models(self):
        """Compare model performance."""
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        comparison = pd.DataFrame({
            model: {
                'Accuracy': results['accuracy'],
                'AUC-ROC': results['auc']
            }
            for model, results in self.results.items()
        }).T
        
        print("\n", comparison)
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {results['auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig('roc_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison
    
    def predict_risk(self, patient_data):
        """Predict cardiovascular risk for new patients."""
        # Scale input data
        patient_scaled = self.scaler.transform(patient_data)
        
        # Get predictions from best model
        best_model = self.models['Random Forest']
        risk_probability = best_model.predict_proba(patient_data)[:, 1]
        risk_class = best_model.predict(patient_data)
        
        return {
            'risk_probability': risk_probability,
            'risk_class': risk_class,
            'risk_level': ['Low' if p < 0.3 else 'Moderate' if p < 0.7 else 'High' 
                          for p in risk_probability]
        }


def main():
    """Main execution function."""
    # Initialize predictor
    predictor = CardiovascularRiskPredictor()
    
    # Load data (using sample data structure)
    # In practice, replace with actual dataset path
    print("Cardiovascular Disease Risk Prediction Model")
    print("="*50)
    print("\nThis model demonstrates machine learning techniques for")
    print("predicting cardiovascular disease risk based on patient data.")
    print("\nKey Features:")
    print("- Logistic Regression with interpretable coefficients")
    print("- Random Forest with feature importance analysis")
    print("- Comprehensive model evaluation and comparison")
    print("- Clinical risk stratification")
    
    # Example usage with sample data
    sample_patient = pd.DataFrame({
        'age_years': [55],
        'gender': [1],
        'height': [170],
        'weight': [75],
        'bmi': [25.9],
        'ap_hi': [140],
        'ap_lo': [90],
        'cholesterol': [2],
        'gluc': [1],
        'smoke': [0],
        'alco': [0],
        'active': [1]
    })
    
    print("\n" + "="*50)
    print("SAMPLE PREDICTION")
    print("="*50)
    print("\nSample Patient Profile:")
    print(f"Age: {sample_patient['age_years'].values[0]} years")
    print(f"BMI: {sample_patient['bmi'].values[0]:.1f}")
    print(f"Blood Pressure: {sample_patient['ap_hi'].values[0]}/{sample_patient['ap_lo'].values[0]}")
    print(f"Cholesterol Level: {sample_patient['cholesterol'].values[0]}")
    
    print("\nNote: To run full analysis, provide path to cardiovascular dataset.")
    print("Expected dataset columns: id, age, gender, height, weight, ap_hi,")
    print("ap_lo, cholesterol, gluc, smoke, alco, active, cardio")


if __name__ == "__main__":
    main()
