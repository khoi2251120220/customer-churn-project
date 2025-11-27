"""
Modeling Module
Huấn luyện và đánh giá các mô hình Machine Learning
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


class ChurnModelTrainer:
    """Class để huấn luyện và đánh giá các mô hình churn prediction"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
    
    def initialize_models(self):
        """Khởi tạo các mô hình để thử nghiệm"""
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
        }
        print(f"Initialized {len(self.models)} models")
        return self.models
    
    def train_all_models(self, X_train, y_train, cv=5):
        """Huấn luyện tất cả các mô hình với cross-validation"""
        print("\n" + "="*60)
        print("Training Models with Cross-Validation")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, 
                                       cv=cv, scoring='roc_auc')
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Store results
            self.results[name] = {
                'model': model,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return self.results
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Đánh giá chi tiết một mô hình"""
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print(f"\n{'='*60}")
        print(f"{model_name} - Evaluation Results")
        print(f"{'='*60}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Classification Report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return metrics, y_pred, y_pred_proba
    
    def evaluate_all_models(self, X_test, y_test):
        """Đánh giá tất cả các mô hình đã train"""
        evaluation_results = {}
        
        for name, result in self.results.items():
            model = result['model']
            metrics, y_pred, y_pred_proba = self.evaluate_model(
                model, X_test, y_test, model_name=name
            )
            evaluation_results[name] = {
                'metrics': metrics,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        
        return evaluation_results
    
    def compare_models(self, evaluation_results):
        """So sánh kết quả các mô hình"""
        comparison_df = pd.DataFrame({
            name: result['metrics'] 
            for name, result in evaluation_results.items()
        }).T
        
        comparison_df = comparison_df.sort_values('roc_auc', ascending=False)
        
        print("\n" + "="*60)
        print("Model Comparison Summary")
        print("="*60)
        print(comparison_df.round(4))
        
        # Select best model
        self.best_model_name = comparison_df.index[0]
        self.best_model = self.results[self.best_model_name]['model']
        
        print(f"\nBest Model: {self.best_model_name}")
        print(f"ROC-AUC Score: {comparison_df.loc[self.best_model_name, 'roc_auc']:.4f}")
        
        return comparison_df
    
    def hyperparameter_tuning(self, model_name, X_train, y_train, param_grid, cv=5):
        """Tinh chỉnh hyperparameters cho một mô hình cụ thể"""
        print(f"\n{'='*60}")
        print(f"Hyperparameter Tuning for {model_name}")
        print(f"{'='*60}")
        
        base_model = self.models[model_name]
        
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, 
            scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        self.results[model_name]['model'] = grid_search.best_estimator_
        self.results[model_name]['best_params'] = grid_search.best_params_
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def plot_roc_curves(self, evaluation_results, y_test, save_path=None):
        """Vẽ ROC curves cho tất cả các mô hình"""
        plt.figure(figsize=(10, 8))
        
        for name, result in evaluation_results.items():
            y_pred_proba = result['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = result['metrics']['roc_auc']
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, y_test, y_pred, model_name, save_path=None):
        """Vẽ confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, model, feature_names, top_n=20, save_path=None):
        """Vẽ feature importance"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.figure(figsize=(10, 8))
            plt.title(f'Top {top_n} Feature Importances')
            plt.barh(range(top_n), importances[indices])
            plt.yticks(range(top_n), [feature_names[i] for i in indices])
            plt.xlabel('Importance')
            plt.gca().invert_yaxis()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
    
    def save_model(self, model, filepath):
        """Lưu mô hình ra file"""
        joblib.dump(model, filepath)
        print(f"\nModel saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load mô hình từ file"""
        model = joblib.load(filepath)
        print(f"\nModel loaded from: {filepath}")
        return model


if __name__ == "__main__":
    print("Modeling module loaded successfully!")
    print("Use this module to train and evaluate churn prediction models.")
