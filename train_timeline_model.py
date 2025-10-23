#!/usr/bin/env python3
"""
Timeline-Based Multi-Agent Communication Prediction Model

This model trains on the complete timeline of all posts and replies,
predicting the next action in the sequence including:
- Who will act next (which agent)
- What type of action (post or reply)
- When they will act (time delay)
- Content characteristics (sentiment, emotion, topic)

CPU-friendly with lightweight text features.
"""

import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
import string
from typing import Dict, List, Tuple, Optional

# ML imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, classification_report
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

# Simple text processing
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

class TimelinePredictor:
    """Timeline-based predictor for multi-agent communication"""
    
    def __init__(self, use_text_features=True, max_history=10):
        self.use_text_features = use_text_features
        self.max_history = max_history
        
        # Models for different predictions
        self.next_agent_model = None
        self.next_action_type_model = None  # post vs reply
        self.next_emotion_model = None
        self.next_topic_model = None
        self.next_sentiment_model = None
        self.time_delay_model = None  # predict when next action happens
        
        # Encoders and scalers
        self.agent_encoder = LabelEncoder()
        self.action_type_encoder = LabelEncoder()
        self.emotion_encoder = LabelEncoder()
        self.topic_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Text analysis
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Vocabulary for simple text features
        self.common_words = set()
        self.question_indicators = ['?', 'what', 'how', 'why', 'when', 'where', 'who']
        self.exclamation_indicators = ['!', 'wow', 'amazing', 'great', 'terrible', 'awful']
        
    def extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract lightweight text features without transformers"""
        if not text or not self.use_text_features:
            return {}
        
        text_lower = text.lower()
        words = text_lower.split()
        
        features = {}
        
        # Basic statistics
        features['text_length'] = len(text)
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['sentence_count'] = len([s for s in text.split('.') if s.strip()])
        
        # Punctuation features
        features['question_marks'] = text.count('?')
        features['exclamation_marks'] = text.count('!')
        features['periods'] = text.count('.')
        features['commas'] = text.count(',')
        features['punctuation_ratio'] = sum(1 for c in text if c in string.punctuation) / len(text) if text else 0
        
        # Capitalization features
        features['capital_letters'] = sum(1 for c in text if c.isupper())
        features['capital_ratio'] = features['capital_letters'] / len(text) if text else 0
        features['all_caps_words'] = sum(1 for w in words if w.isupper() and len(w) > 1)
        
        # Question and exclamation indicators
        features['question_words'] = sum(1 for indicator in self.question_indicators if indicator in text_lower)
        features['exclamation_words'] = sum(1 for indicator in self.exclamation_indicators if indicator in text_lower)
        
        # Repetition features
        features['repeated_chars'] = len(re.findall(r'(.)\1{2,}', text))
        features['repeated_words'] = len(words) - len(set(words)) if words else 0
        
        # Sentiment features (VADER is lightweight)
        try:
            sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
            features['sentiment_compound'] = sentiment_scores['compound']
            features['sentiment_positive'] = sentiment_scores['pos']
            features['sentiment_negative'] = sentiment_scores['neg']
            features['sentiment_neutral'] = sentiment_scores['neu']
        except:
            features['sentiment_compound'] = 0.0
            features['sentiment_positive'] = 0.0
            features['sentiment_negative'] = 0.0
            features['sentiment_neutral'] = 1.0
        
        # Common word features (if vocabulary is built)
        if self.common_words:
            features['common_word_count'] = sum(1 for w in words if w in self.common_words)
            features['common_word_ratio'] = features['common_word_count'] / len(words) if words else 0
        else:
            features['common_word_count'] = 0
            features['common_word_ratio'] = 0
        
        return features
    
    def build_vocabulary(self, all_items: List[Dict]):
        """Build a simple vocabulary of common words"""
        all_words = []
        for item in all_items:
            if 'post' in item and item['post']:
                words = item['post'].lower().split()
                all_words.extend(words)
            if 'reply' in item and item.get('reply'):
                words = item['reply'].lower().split()
                all_words.extend(words)
        
        # Get top 1000 most common words
        word_counts = Counter(all_words)
        self.common_words = set([word for word, count in word_counts.most_common(1000)])
        print(f"Built vocabulary with {len(self.common_words)} common words")
    
    def create_timeline(self, all_items: List[Dict]) -> List[Dict]:
        """Create chronological timeline from all posts and replies"""
        print("Creating chronological timeline from all posts...")
        timeline = []
        
        for item_idx, item in enumerate(all_items):
            if item_idx % 5000 == 0:
                print(f"Processing item {item_idx}/{len(all_items)}")
            
            # Extract post action
            if 'post' in item and item['post']:
                try:
                    if 'post_time' in item and item['post_time']:
                        post_time = datetime.strptime(item['post_time'], "%A %B %d %Y, %H:%M:%S")
                    else:
                        post_time = datetime(2019, 4, 30, 22, 0, 0) + timedelta(minutes=item_idx)
                except:
                    post_time = datetime(2019, 4, 30, 22, 0, 0) + timedelta(minutes=item_idx)
                
                timeline.append({
                    'timestamp': post_time,
                    'agent': item.get('agent_type', 'unknown'),
                    'action_type': 'post',
                    'content': item['post'],
                    'emotion': 'neutral',  # We'll predict this
                    'topic': 'general',   # We'll predict this
                    'original_index': item_idx
                })
                
                # Extract reply action if exists
                if 'reply' in item and item['reply']:
                    try:
                        if 'reply_time' in item and item['reply_time']:
                            reply_time = datetime.strptime(item['reply_time'], "%A %B %d %Y, %H:%M:%S")
                        else:
                            reply_time = post_time + timedelta(seconds=1)
                    except:
                        reply_time = post_time + timedelta(seconds=1)
                    
                    timeline.append({
                        'timestamp': reply_time,
                        'agent': item.get('agent_type', 'unknown'),
                        'action_type': 'reply',
                        'content': item['reply'],
                        'emotion': 'neutral',
                        'topic': 'general',
                        'original_index': item_idx
                    })
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x['timestamp'])
        print(f"Created timeline with {len(timeline)} total actions")
        return timeline
    
    def extract_timeline_features(self, recent_history: List[Dict], context: Dict) -> np.ndarray:
        """Extract features from timeline history"""
        features = []
        
        # Basic timeline context
        features.extend([
            context.get('time_since_start', 0),
            context.get('time_since_last_action', 0),
            context.get('total_actions_so_far', 0),
            context.get('recent_actions_count', 0),
            context.get('recent_posts_count', 0),
            context.get('recent_replies_count', 0),
            context.get('unique_agents_recent', 0),
        ])
        
        # Agent activity patterns in recent history
        if recent_history:
            # Last action features
            last_action = recent_history[-1]
            
            # Encode last agent (handle unknown agents)
            try:
                last_agent_encoded = self.agent_encoder.transform([last_action['agent']])[0]
            except:
                last_agent_encoded = 0
            features.append(last_agent_encoded)
            
            # Last action type
            try:
                last_action_type_encoded = self.action_type_encoder.transform([last_action['action_type']])[0]
            except:
                last_action_type_encoded = 0
            features.append(last_action_type_encoded)
            
            # Text features from last action
            if self.use_text_features and 'content' in last_action:
                text_features = self.extract_text_features(last_action['content'])
                text_feature_names = [
                    'text_length', 'word_count', 'avg_word_length', 'sentence_count',
                    'question_marks', 'exclamation_marks', 'periods', 'commas', 'punctuation_ratio',
                    'capital_letters', 'capital_ratio', 'all_caps_words',
                    'question_words', 'exclamation_words', 'repeated_chars', 'repeated_words',
                    'sentiment_compound', 'sentiment_positive', 'sentiment_negative', 'sentiment_neutral',
                    'common_word_count', 'common_word_ratio'
                ]
                for feature_name in text_feature_names:
                    features.append(text_features.get(feature_name, 0))
            else:
                features.extend([0] * 22)
            
            # Agent activity counts in recent history
            agent_counts = Counter([action['agent'] for action in recent_history])
            most_active_agent_count = max(agent_counts.values()) if agent_counts else 0
            features.append(most_active_agent_count)
            
            # Action type ratios in recent history
            action_counts = Counter([action['action_type'] for action in recent_history])
            post_ratio = action_counts.get('post', 0) / len(recent_history) if recent_history else 0
            reply_ratio = action_counts.get('reply', 0) / len(recent_history) if recent_history else 0
            features.extend([post_ratio, reply_ratio])
            
            # Average sentiment in recent history
            if self.use_text_features:
                recent_sentiments = []
                for action in recent_history:
                    if 'content' in action:
                        text_features = self.extract_text_features(action['content'])
                        recent_sentiments.append(text_features.get('sentiment_compound', 0))
                features.append(np.mean(recent_sentiments) if recent_sentiments else 0)
            else:
                features.append(0)
        else:
            # No history - add zeros
            features.extend([0, 0])  # last agent, last action type
            features.extend([0] * 22)  # text features
            features.extend([0, 0, 0, 0])  # agent activity, post/reply ratios, avg sentiment
        
        # Time-based features
        features.extend([
            context.get('hour_of_day', 12),
            context.get('day_of_week', 1),
            context.get('is_weekend', 0),
        ])
        
        return np.array(features, dtype=np.float32)
    
    def prepare_training_data(self, all_items: List[Dict]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare training data from timeline"""
        # Build vocabulary first
        self.build_vocabulary(all_items)
        
        # Create timeline
        timeline = self.create_timeline(all_items)
        
        if len(timeline) < 2:
            raise ValueError("Timeline too short for training")
        
        print("Extracting features from timeline...")
        X = []
        y_agent = []
        y_action_type = []
        y_emotion = []
        y_topic = []
        y_sentiment = []
        y_time_delay = []
        
        all_agents = set()
        all_action_types = set()
        all_emotions = set()
        all_topics = set()
        
        timeline_start = timeline[0]['timestamp']
        
        # Create training pairs from consecutive actions in timeline
        for i in range(1, len(timeline)):
            if i % 1000 == 0:
                print(f"Processing timeline position {i}/{len(timeline)}")
            
            current_action = timeline[i]
            history = timeline[:i]
            
            # Get recent history (last 10 actions)
            recent_history = history[-self.max_history:] if len(history) >= self.max_history else history
            
            # Calculate time-based features
            current_time = current_action['timestamp']
            time_since_start = (current_time - timeline_start).total_seconds() / 3600  # hours
            
            # Time since last action
            if history:
                last_action_time = history[-1]['timestamp']
                time_since_last = (current_time - last_action_time).total_seconds() / 60  # minutes
                time_since_last = min(time_since_last, 1440)  # Cap at 24 hours
            else:
                time_since_last = 0
            
            # Agent activity in recent history
            recent_agent_counts = Counter([action['agent'] for action in recent_history])
            unique_agents_recent = len(recent_agent_counts)
            
            # Action type counts in recent history
            recent_action_counts = Counter([action['action_type'] for action in recent_history])
            recent_posts = recent_action_counts.get('post', 0)
            recent_replies = recent_action_counts.get('reply', 0)
            
            context = {
                'time_since_start': time_since_start,
                'time_since_last_action': time_since_last,
                'total_actions_so_far': len(history),
                'recent_actions_count': len(recent_history),
                'recent_posts_count': recent_posts,
                'recent_replies_count': recent_replies,
                'unique_agents_recent': unique_agents_recent,
                'hour_of_day': current_time.hour,
                'day_of_week': current_time.weekday(),
                'is_weekend': 1 if current_time.weekday() >= 5 else 0,
            }
            
            # Extract features
            features = self.extract_timeline_features(recent_history, context)
            X.append(features)
            
            # Extract targets
            y_agent.append(current_action['agent'])
            y_action_type.append(current_action['action_type'])
            y_emotion.append(current_action['emotion'])
            y_topic.append(current_action['topic'])
            y_time_delay.append(time_since_last)
            
            # Calculate sentiment from content
            if 'content' in current_action and current_action['content']:
                text_features = self.extract_text_features(current_action['content'])
                y_sentiment.append(text_features.get('sentiment_compound', 0))
            else:
                y_sentiment.append(0)
            
            # Collect unique values for encoders
            all_agents.add(current_action['agent'])
            all_action_types.add(current_action['action_type'])
            all_emotions.add(current_action['emotion'])
            all_topics.add(current_action['topic'])
        
        # Fit encoders
        print("Fitting encoders...")
        self.agent_encoder.fit(list(all_agents))
        self.action_type_encoder.fit(list(all_action_types))
        self.emotion_encoder.fit(list(all_emotions))
        self.topic_encoder.fit(list(all_topics))
        
        # Convert to arrays
        if not X:
            raise ValueError("No training data extracted")
        
        X = np.array(X)
        print(f"Extracted {len(X)} samples with {X.shape[1]} features each")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        targets = {
            'agent': self.agent_encoder.transform(y_agent),
            'action_type': self.action_type_encoder.transform(y_action_type),
            'emotion': self.emotion_encoder.transform(y_emotion),
            'topic': self.topic_encoder.transform(y_topic),
            'sentiment': np.array(y_sentiment, dtype=np.float32),
            'time_delay': np.array(y_time_delay, dtype=np.float32)
        }
        
        print(f"Prepared {len(X_scaled)} training samples")
        print(f"Agent distribution: {Counter(y_agent)}")
        print(f"Action type distribution: {Counter(y_action_type)}")
        print(f"Average time delay: {np.mean(y_time_delay):.2f} minutes")
        
        return X_scaled, targets
    
    def train(self, all_items: List[Dict]):
        """Train all models"""
        print("Preparing training data...")
        X, targets = self.prepare_training_data(all_items)
        
        print("Training models...")
        
        # Calculate class weights for imbalanced data
        agent_classes = np.unique(targets['agent'])
        agent_weights = compute_class_weight('balanced', classes=agent_classes, y=targets['agent'])
        agent_weight_dict = dict(zip(agent_classes, agent_weights))
        
        action_type_classes = np.unique(targets['action_type'])
        action_type_weights = compute_class_weight('balanced', classes=action_type_classes, y=targets['action_type'])
        action_type_weight_dict = dict(zip(action_type_classes, action_type_weights))
        
        # Train next agent model
        print("Training next agent model...")
        self.next_agent_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            class_weight=agent_weight_dict,
            n_jobs=-1
        )
        self.next_agent_model.fit(X, targets['agent'])
        
        # Train action type model (post vs reply)
        print("Training action type model...")
        self.next_action_type_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            class_weight=action_type_weight_dict,
            n_jobs=-1
        )
        self.next_action_type_model.fit(X, targets['action_type'])
        
        # Train emotion model
        print("Training emotion model...")
        self.next_emotion_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.next_emotion_model.fit(X, targets['emotion'])
        
        # Train topic model
        print("Training topic model...")
        self.next_topic_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.next_topic_model.fit(X, targets['topic'])
        
        # Train sentiment model (regression)
        print("Training sentiment model...")
        self.next_sentiment_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.next_sentiment_model.fit(X, targets['sentiment'])
        
        # Train time delay model (regression)
        print("Training time delay model...")
        self.time_delay_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.time_delay_model.fit(X, targets['time_delay'])
        
        print("Training completed!")
    
    def predict(self, recent_history: List[Dict], current_context: Dict) -> Dict:
        """Make predictions for next action"""
        features = self.extract_timeline_features(recent_history, current_context)
        features_scaled = self.scaler.transform([features])
        
        # Get predictions
        next_agent_encoded = self.next_agent_model.predict(features_scaled)[0]
        next_action_type_encoded = self.next_action_type_model.predict(features_scaled)[0]
        next_emotion_encoded = self.next_emotion_model.predict(features_scaled)[0]
        next_topic_encoded = self.next_topic_model.predict(features_scaled)[0]
        next_sentiment = self.next_sentiment_model.predict(features_scaled)[0]
        time_delay = self.time_delay_model.predict(features_scaled)[0]
        
        # Get probabilities
        agent_probs = self.next_agent_model.predict_proba(features_scaled)[0]
        action_type_probs = self.next_action_type_model.predict_proba(features_scaled)[0]
        emotion_probs = self.next_emotion_model.predict_proba(features_scaled)[0]
        topic_probs = self.next_topic_model.predict_proba(features_scaled)[0]
        
        return {
            'next_agent': self.agent_encoder.inverse_transform([next_agent_encoded])[0],
            'next_action_type': self.action_type_encoder.inverse_transform([next_action_type_encoded])[0],
            'next_emotion': self.emotion_encoder.inverse_transform([next_emotion_encoded])[0],
            'next_topic': self.topic_encoder.inverse_transform([next_topic_encoded])[0],
            'next_sentiment': float(next_sentiment),
            'time_delay_minutes': max(0, float(time_delay)),
            'probabilities': {
                'agent': dict(zip(self.agent_encoder.classes_, agent_probs)),
                'action_type': dict(zip(self.action_type_encoder.classes_, action_type_probs)),
                'emotion': dict(zip(self.emotion_encoder.classes_, emotion_probs)),
                'topic': dict(zip(self.topic_encoder.classes_, topic_probs)),
            },
            'confidence': {
                'agent': float(np.max(agent_probs)),
                'action_type': float(np.max(action_type_probs)),
                'emotion': float(np.max(emotion_probs)),
                'topic': float(np.max(topic_probs)),
                'sentiment': 1.0 - abs(next_sentiment) if abs(next_sentiment) <= 1 else 0.5,
                'time_delay': 1.0 / (1.0 + abs(time_delay) / 60)  # Higher confidence for shorter delays
            }
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'next_agent_model': self.next_agent_model,
            'next_action_type_model': self.next_action_type_model,
            'next_emotion_model': self.next_emotion_model,
            'next_topic_model': self.next_topic_model,
            'next_sentiment_model': self.next_sentiment_model,
            'time_delay_model': self.time_delay_model,
            'agent_encoder': self.agent_encoder,
            'action_type_encoder': self.action_type_encoder,
            'emotion_encoder': self.emotion_encoder,
            'topic_encoder': self.topic_encoder,
            'scaler': self.scaler,
            'common_words': self.common_words,
            'use_text_features': self.use_text_features,
            'max_history': self.max_history,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.next_agent_model = model_data['next_agent_model']
        self.next_action_type_model = model_data['next_action_type_model']
        self.next_emotion_model = model_data['next_emotion_model']
        self.next_topic_model = model_data['next_topic_model']
        self.next_sentiment_model = model_data['next_sentiment_model']
        self.time_delay_model = model_data['time_delay_model']
        self.agent_encoder = model_data['agent_encoder']
        self.action_type_encoder = model_data['action_type_encoder']
        self.emotion_encoder = model_data['emotion_encoder']
        self.topic_encoder = model_data['topic_encoder']
        self.scaler = model_data['scaler']
        self.common_words = model_data.get('common_words', set())
        self.use_text_features = model_data.get('use_text_features', True)
        self.max_history = model_data.get('max_history', 10)
        print(f"Model loaded from {filepath}")

def evaluate_model(model, all_items, test_size=0.2):
    """Evaluate the model performance"""
    print("Evaluating model...")
    
    # Split data
    train_items, test_items = train_test_split(all_items, test_size=test_size, random_state=42)
    
    # Create a temporary model for test data
    test_model = TimelinePredictor(model.use_text_features, model.max_history)
    X_test, y_test = test_model.prepare_training_data(test_items)
    
    # Copy encoders from trained model
    test_model.agent_encoder = model.agent_encoder
    test_model.action_type_encoder = model.action_type_encoder
    test_model.emotion_encoder = model.emotion_encoder
    test_model.topic_encoder = model.topic_encoder
    test_model.scaler = model.scaler
    
    # Make predictions
    y_pred = {}
    y_pred['agent'] = model.next_agent_model.predict(X_test)
    y_pred['action_type'] = model.next_action_type_model.predict(X_test)
    y_pred['emotion'] = model.next_emotion_model.predict(X_test)
    y_pred['topic'] = model.next_topic_model.predict(X_test)
    y_pred['sentiment'] = model.next_sentiment_model.predict(X_test)
    y_pred['time_delay'] = model.time_delay_model.predict(X_test)
    
    # Calculate metrics
    results = {}
    
    # Classification metrics
    for task in ['agent', 'action_type', 'emotion', 'topic']:
        results[task] = {
            'accuracy': accuracy_score(y_test[task], y_pred[task]),
            'f1_score': f1_score(y_test[task], y_pred[task], average='weighted'),
        }
    
    # Regression metrics
    for task in ['sentiment', 'time_delay']:
        results[task] = {
            'mae': mean_absolute_error(y_test[task], y_pred[task]),
            'mse': np.mean((y_test[task] - y_pred[task]) ** 2)
        }
    
    return results

def main():
    """Main training function"""
    print("Starting Timeline-Based Model Training...")
    
    # Load data
    data_file = "data/full_dataset_for_generation/enhanced_sorted_temporal_readable_88330_items_20250810.jsonl"
    
    print(f"Loading data from {data_file}...")
    all_items = []
    with open(data_file, 'r') as f:
        for line in f:
            all_items.append(json.loads(line))
    
    print(f"Loaded {len(all_items)} items")
    
    # Create and train model
    model = TimelinePredictor(use_text_features=True, max_history=10)
    model.train(all_items)
    
    # Evaluate model
    print("\nEvaluating model...")
    results = evaluate_model(model, all_items, test_size=0.1)  # Use smaller test set for speed
    print("\nEvaluation Results:")
    for task, metrics in results.items():
        print(f"{task.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Save model
    model_path = "model_export/timeline_model.pkl"
    model.save_model(model_path)
    
    # Save evaluation results
    stats = {
        'model_type': 'Timeline-Based Multi-Agent Predictor',
        'training_items': len(all_items),
        'features': 'Timeline sequence + lightweight text features',
        'predictions': ['next_agent', 'action_type', 'emotion', 'topic', 'sentiment', 'time_delay'],
        'evaluation_results': results,
        'model_path': model_path,
        'training_date': datetime.now().isoformat()
    }
    
    with open('model_export/timeline_model_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nTimeline model and statistics saved!")
    print("This model can predict:")
    print("- Which agent will act next")
    print("- Whether they will post or reply")
    print("- When they will act (time delay)")
    print("- Content characteristics (sentiment, emotion, topic)")

if __name__ == "__main__":
    main()
