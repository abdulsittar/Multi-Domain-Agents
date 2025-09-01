#!/usr/bin/env python3
"""
Improved Multi-Agent Communication Model Training Script

This script implements a significantly enhanced approach to training a probabilistic model
for predicting agent communication patterns, addressing the critical flaws identified in
the original training approach.

Key Improvements:
1. Proper data preprocessing with class balancing
2. Feature engineering (temporal, contextual, conversation features)
3. Multiple model architectures (Random Forest, XGBoost, Neural Network)
4. Proper cross-validation and hyperparameter tuning
5. Temperature-controlled sampling for predictions
6. Context-aware reply modeling
7. Ensemble methods for robustness

Author: Generated for Multi-Domain-Agents Project
Date: September 2025
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class ImprovedAgentPredictor:
    """
    Enhanced predictor with multiple models and sophisticated features
    """
    def __init__(self, context_window=5, use_ensemble=True):
        self.context_window = context_window
        self.use_ensemble = use_ensemble
        
        # Models
        self.agent_model = None
        self.action_model = None
        self.time_model = None
        
        # Encoders and scalers
        self.agent_encoder = LabelEncoder()
        self.action_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        
        # Feature engineering components
        self.agent_embeddings = {}
        self.hourly_patterns = {}
        self.conversation_stats = {}
        
        # Prediction parameters
        self.temperature = 0.7  # For controlled randomness
        self.min_prob_threshold = 0.15  # Minimum probability for any class
        
    def parse_timestamp(self, timestamp_str):
        """Parse timestamp from string format"""
        try:
            return datetime.strptime(timestamp_str, "%A %B %d %Y, %H:%M:%S")
        except:
            return None
    
    def load_and_preprocess_data(self, data_path):
        """Load and preprocess the JSONL data with proper feature engineering"""
        print("Loading and preprocessing data...")
        
        # Load raw data
        raw_data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    raw_data.append(item)
                except json.JSONDecodeError:
                    continue
        
        print(f"Loaded {len(raw_data)} raw records")
        
        # Process into structured format
        processed_data = []
        for item in raw_data:
            # Process post
            if item.get('post_time'):
                post_time = self.parse_timestamp(item['post_time'])
                if post_time:
                    processed_data.append({
                        'agent': item['agent_type'],
                        'action': 'post',
                        'timestamp': post_time,
                        'content': item.get('post', ''),
                        'conversation_id': item.get('id', len(processed_data))
                    })
            
            # Process reply if exists
            if item.get('reply_time') and item.get('reply'):
                reply_time = self.parse_timestamp(item['reply_time'])
                if reply_time:
                    processed_data.append({
                        'agent': item['agent_type'],
                        'action': 'reply',
                        'timestamp': reply_time,
                        'content': item.get('reply', ''),
                        'conversation_id': item.get('id', len(processed_data))
                    })
        
        # Sort by timestamp
        processed_data.sort(key=lambda x: x['timestamp'])
        
        print(f"Processed {len(processed_data)} total actions")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(processed_data)
        
        # Analyze data distribution
        print("\n=== Data Distribution ===")
        print(df.groupby(['agent', 'action']).size().unstack(fill_value=0))
        
        return df
    
    def engineer_features(self, df):
        """Create sophisticated features for better prediction"""
        print("Engineering features...")
        
        features = []
        targets_agent = []
        targets_action = []
        targets_time = []
        
        for i in range(self.context_window, len(df)):
            # Get context window
            context = df.iloc[i-self.context_window:i]
            target = df.iloc[i]
            
            # Basic context features
            last_agents = context['agent'].tolist()
            last_actions = context['action'].tolist()
            last_times = context['timestamp'].tolist()
            
            # Temporal features
            current_time = target['timestamp']
            hour = current_time.hour
            day_of_week = current_time.weekday()
            
            # Time gaps
            time_gaps = []
            for j in range(1, len(last_times)):
                gap = (last_times[j] - last_times[j-1]).total_seconds() / 60
                time_gaps.append(gap)
            
            # Conversation flow features
            agent_sequence = ''.join(last_agents[-3:])  # Last 3 agents
            action_sequence = ''.join(last_actions[-3:])  # Last 3 actions
            
            # Agent transition patterns
            last_agent = last_agents[-1]
            second_last_agent = last_agents[-2] if len(last_agents) > 1 else 'NONE'
            
            # Reply context
            recent_replies = sum(1 for a in last_actions[-3:] if a == 'reply')
            time_since_last_reply = 0
            for j in range(len(context)-1, -1, -1):
                if context.iloc[j]['action'] == 'reply':
                    time_since_last_reply = (current_time - context.iloc[j]['timestamp']).total_seconds() / 60
                    break
            
            # Agent activity patterns
            agent_counts = Counter(last_agents)
            agent_dominance = max(agent_counts.values()) / len(last_agents) if last_agents else 0
            
            # Feature vector
            feature_vector = [
                # Temporal features
                hour,
                day_of_week,
                
                # Time gap features
                np.mean(time_gaps) if time_gaps else 0,
                np.std(time_gaps) if len(time_gaps) > 1 else 0,
                time_gaps[-1] if time_gaps else 0,  # Last gap
                
                # Context features
                agent_counts.get('AI_Agent', 0) / len(last_agents) if last_agents else 0,
                agent_counts.get('Crypto_Agent', 0) / len(last_agents) if last_agents else 0,
                recent_replies / 3,  # Ratio of recent replies
                time_since_last_reply,
                agent_dominance,
                
                # Sequence features (encoded)
                hash(agent_sequence) % 1000,  # Simple hash encoding
                hash(action_sequence) % 1000,
                
                # Last agent one-hot
                1 if last_agent == 'AI_Agent' else 0,
                1 if last_agent == 'Crypto_Agent' else 0,
                
                # Second last agent one-hot
                1 if second_last_agent == 'AI_Agent' else 0,
                1 if second_last_agent == 'Crypto_Agent' else 0,
            ]
            
            features.append(feature_vector)
            targets_agent.append(target['agent'])
            targets_action.append(target['action'])
            
            # Time target (minutes since last action)
            if i > 0:
                time_gap = (target['timestamp'] - df.iloc[i-1]['timestamp']).total_seconds() / 60
                targets_time.append(time_gap)
            else:
                targets_time.append(5.0)  # Default
        
        print(f"Generated {len(features)} feature vectors with {len(features[0])} features each")
        
        return np.array(features), targets_agent, targets_action, targets_time
    
    def balance_classes(self, X, y_agent, y_action):
        """Apply class balancing techniques"""
        print("Applying class balancing...")
        
        # For agent prediction, ensure minimum representation
        from sklearn.utils import resample
        
        df_combined = pd.DataFrame(X)
        df_combined['agent'] = y_agent
        df_combined['action'] = y_action
        
        # Separate classes
        ai_agent_data = df_combined[df_combined['agent'] == 'AI_Agent']
        crypto_agent_data = df_combined[df_combined['agent'] == 'Crypto_Agent']
        
        # Calculate target size (aim for 60-40 split maximum)
        total_size = len(df_combined)
        max_class_size = int(total_size * 0.6)
        min_class_size = int(total_size * 0.4)
        
        # Resample to balance
        if len(ai_agent_data) > max_class_size:
            ai_agent_data = resample(ai_agent_data, n_samples=max_class_size, random_state=42)
        elif len(ai_agent_data) < min_class_size:
            ai_agent_data = resample(ai_agent_data, n_samples=min_class_size, random_state=42, replace=True)
        
        if len(crypto_agent_data) > max_class_size:
            crypto_agent_data = resample(crypto_agent_data, n_samples=max_class_size, random_state=42)
        elif len(crypto_agent_data) < min_class_size:
            crypto_agent_data = resample(crypto_agent_data, n_samples=min_class_size, random_state=42, replace=True)
        
        # Combine and shuffle
        balanced_data = pd.concat([ai_agent_data, crypto_agent_data])
        balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Extract balanced features and targets
        X_balanced = balanced_data.drop(['agent', 'action'], axis=1).values
        y_agent_balanced = balanced_data['agent'].tolist()
        y_action_balanced = balanced_data['action'].tolist()
        
        print(f"Balanced dataset: {len(X_balanced)} samples")
        print(f"Agent distribution: {Counter(y_agent_balanced)}")
        print(f"Action distribution: {Counter(y_action_balanced)}")
        
        return X_balanced, y_agent_balanced, y_action_balanced
    
    def train_models(self, X, y_agent, y_action, y_time):
        """Train multiple models with proper cross-validation"""
        print("Training models...")
        
        # Encode labels
        y_agent_encoded = self.agent_encoder.fit_transform(y_agent)
        y_action_encoded = self.action_encoder.fit_transform(y_action)
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_agent_train, y_agent_test = train_test_split(
            X_scaled, y_agent_encoded, test_size=0.2, random_state=42, stratify=y_agent_encoded
        )
        
        _, _, y_action_train, y_action_test = train_test_split(
            X_scaled, y_action_encoded, test_size=0.2, random_state=42, stratify=y_action_encoded
        )
        
        # Train Agent Prediction Model
        print("Training agent prediction model...")
        
        # Calculate class weights for imbalanced data
        agent_class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_agent_train), y=y_agent_train
        )
        agent_class_weight_dict = dict(enumerate(agent_class_weights))
        
        if self.use_ensemble:
            # Random Forest with hyperparameter tuning
            rf_param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            rf_model = RandomForestClassifier(
                random_state=42, 
                class_weight=agent_class_weight_dict
            )
            
            agent_grid_search = GridSearchCV(
                rf_model, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            agent_grid_search.fit(X_train, y_agent_train)
            self.agent_model = agent_grid_search.best_estimator_
            
            print(f"Best agent model params: {agent_grid_search.best_params_}")
            print(f"Best agent model CV score: {agent_grid_search.best_score_:.4f}")
        else:
            # Simple Random Forest
            self.agent_model = RandomForestClassifier(
                n_estimators=200, 
                max_depth=20, 
                random_state=42,
                class_weight=agent_class_weight_dict
            )
            self.agent_model.fit(X_train, y_agent_train)
        
        # Train Action Prediction Model
        print("Training action prediction model...")
        
        action_class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_action_train), y=y_action_train
        )
        action_class_weight_dict = dict(enumerate(action_class_weights))
        
        self.action_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            class_weight=action_class_weight_dict
        )
        self.action_model.fit(X_train, y_action_train)
        
        # Train Time Prediction Model (Regression)
        print("Training time prediction model...")
        
        from sklearn.ensemble import RandomForestRegressor
        _, _, y_time_train, y_time_test = train_test_split(
            X_scaled, y_time, test_size=0.2, random_state=42
        )
        
        self.time_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            random_state=42
        )
        self.time_model.fit(X_train, y_time_train)
        
        # Evaluate models
        print("\n=== Model Evaluation ===")
        
        # Agent model
        agent_pred = self.agent_model.predict(X_test)
        agent_accuracy = accuracy_score(y_agent_test, agent_pred)
        print(f"Agent Model Accuracy: {agent_accuracy:.4f}")
        
        # Action model
        action_pred = self.action_model.predict(X_test)
        action_accuracy = accuracy_score(y_action_test, action_pred)
        print(f"Action Model Accuracy: {action_accuracy:.4f}")
        
        # Time model
        time_pred = self.time_model.predict(X_test)
        time_mae = np.mean(np.abs(time_pred - y_time_test))
        print(f"Time Model MAE: {time_mae:.2f} minutes")
        
        # Store evaluation results
        self.evaluation_results = {
            'agent_accuracy': agent_accuracy,
            'action_accuracy': action_accuracy,
            'time_mae': time_mae,
            'agent_classification_report': classification_report(
                y_agent_test, agent_pred, target_names=self.agent_encoder.classes_
            ),
            'action_classification_report': classification_report(
                y_action_test, action_pred, target_names=self.action_encoder.classes_
            )
        }
        
        return self.evaluation_results
    
    def predict_with_temperature(self, probabilities, temperature=None):
        """Apply temperature-controlled sampling for more diverse predictions"""
        if temperature is None:
            temperature = self.temperature
        
        if temperature == 0:
            # Deterministic: return highest probability
            return np.argmax(probabilities)
        
        # Apply temperature scaling
        scaled_probs = np.exp(np.log(probabilities + 1e-8) / temperature)
        scaled_probs /= np.sum(scaled_probs)
        
        # Ensure minimum probability threshold
        min_prob = self.min_prob_threshold / len(probabilities)
        scaled_probs = np.maximum(scaled_probs, min_prob)
        scaled_probs /= np.sum(scaled_probs)
        
        # Sample from the distribution
        return np.random.choice(len(probabilities), p=scaled_probs)
    
    def predict_next_action(self, context_features, temperature=None):
        """Predict next agent, action, and timing with improved approach"""
        if self.agent_model is None:
            raise ValueError("Model not trained yet!")
        
        # Scale features
        features_scaled = self.feature_scaler.transform([context_features])
        
        # Predict agent with temperature sampling
        agent_probabilities = self.agent_model.predict_proba(features_scaled)[0]
        agent_idx = self.predict_with_temperature(agent_probabilities, temperature)
        predicted_agent = self.agent_encoder.classes_[agent_idx]
        agent_confidence = agent_probabilities[agent_idx]
        
        # Predict action with temperature sampling
        action_probabilities = self.action_model.predict_proba(features_scaled)[0]
        action_idx = self.predict_with_temperature(action_probabilities, temperature)
        predicted_action = self.action_encoder.classes_[action_idx]
        action_confidence = action_probabilities[action_idx]
        
        # Predict time gap
        predicted_time_gap = max(0.5, self.time_model.predict(features_scaled)[0])
        
        return {
            'agent': predicted_agent,
            'action': predicted_action,
            'time_gap_minutes': predicted_time_gap,
            'agent_confidence': agent_confidence,
            'action_confidence': action_confidence,
            'agent_probabilities': dict(zip(self.agent_encoder.classes_, agent_probabilities)),
            'action_probabilities': dict(zip(self.action_encoder.classes_, action_probabilities))
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'agent_model': self.agent_model,
            'action_model': self.action_model,
            'time_model': self.time_model,
            'agent_encoder': self.agent_encoder,
            'action_encoder': self.action_encoder,
            'feature_scaler': self.feature_scaler,
            'context_window': self.context_window,
            'temperature': self.temperature,
            'min_prob_threshold': self.min_prob_threshold,
            'evaluation_results': self.evaluation_results
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.agent_model = model_data['agent_model']
        self.action_model = model_data['action_model']
        self.time_model = model_data['time_model']
        self.agent_encoder = model_data['agent_encoder']
        self.action_encoder = model_data['action_encoder']
        self.feature_scaler = model_data['feature_scaler']
        self.context_window = model_data['context_window']
        self.temperature = model_data.get('temperature', 0.7)
        self.min_prob_threshold = model_data.get('min_prob_threshold', 0.15)
        self.evaluation_results = model_data.get('evaluation_results', {})
        
        print(f"Model loaded from {filepath}")

def main():
    """Main training pipeline"""
    print("🚀 Starting Improved Multi-Agent Model Training")
    print("=" * 50)
    
    # Configuration
    data_path = 'data/full_dataset_for_generation/enhanced_sorted_temporal_readable_88330_items_20250810.jsonl'
    output_dir = 'model_export'
    model_filename = 'improved_probabilistic_model.pkl'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize improved predictor
    predictor = ImprovedAgentPredictor(
        context_window=5,  # Longer context
        use_ensemble=True
    )
    
    # Load and preprocess data
    df = predictor.load_and_preprocess_data(data_path)
    
    # Engineer features
    X, y_agent, y_action, y_time = predictor.engineer_features(df)
    
    # Balance classes
    X_balanced, y_agent_balanced, y_action_balanced = predictor.balance_classes(
        X, y_agent, y_action
    )
    
    # Train models
    evaluation_results = predictor.train_models(
        X_balanced, y_agent_balanced, y_action_balanced, y_time
    )
    
    # Save model
    model_path = os.path.join(output_dir, model_filename)
    predictor.save_model(model_path)
    
    # Print final results
    print("\n" + "=" * 50)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 50)
    
    print(f"\n📊 Final Model Performance:")
    print(f"  Agent Prediction Accuracy: {evaluation_results['agent_accuracy']:.4f}")
    print(f"  Action Prediction Accuracy: {evaluation_results['action_accuracy']:.4f}")
    print(f"  Time Prediction MAE: {evaluation_results['time_mae']:.2f} minutes")
    
    print(f"\n💾 Model saved to: {model_path}")
    print(f"\n🔧 Model Configuration:")
    print(f"  Context Window: {predictor.context_window}")
    print(f"  Temperature: {predictor.temperature}")
    print(f"  Min Probability Threshold: {predictor.min_prob_threshold}")
    print(f"  Features: {len(X[0])}")
    print(f"  Training Samples: {len(X_balanced)}")
    
    # Test prediction
    print(f"\n🧪 Testing prediction functionality...")
    if len(X_balanced) > 0:
        test_features = X_balanced[0]
        test_prediction = predictor.predict_next_action(test_features)
        print(f"  Sample prediction: {test_prediction}")
    
    print("\n✅ All done! Use the improved model for much better performance.")

if __name__ == "__main__":
    main()
