#!/usr/bin/env python3
"""
Training Script for Your Specific JSONL Format
===============================================
This script is tailored for your data format with 88,330 items.
"""

import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class AgentTimeProbabilisticModel:
    """Probabilistic model for your social network data."""
    
    def __init__(self, context_window: int = 3):
        self.context_window = context_window
        self.data_stats = {}
        
    def load_and_process_data(self, filepath: str) -> pd.DataFrame:
        """Load your specific JSONL format."""
        
        logger.info(f"Loading data from {filepath}...")
        data = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 10000 == 0:
                    logger.info(f"  Processed {line_num} lines...")
                
                record = json.loads(line.strip())
                
                # Extract timestamp
                timestamp_str = record.get('post_time') or record.get('reply_time', '')
                if timestamp_str:
                    timestamp = datetime.strptime(timestamp_str, "%A %B %d %Y, %H:%M:%S")
                else:
                    continue
                
                # Determine action and content
                if record.get('reply', ''):
                    action = 'reply'
                    content = record['reply']
                else:
                    action = 'post'
                    content = record['post']
                
                data.append({
                    'agent': record['agent_type'],
                    'action': action,
                    'content': content,
                    'timestamp': timestamp,
                    'hour': timestamp.hour,
                    'day_of_week': timestamp.weekday(),
                    'minute': timestamp.minute
                })
        
        logger.info(f"Loaded {len(data)} valid records")
        
        # Convert to DataFrame and sort by time
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate time gaps
        df['time_since_last'] = df['timestamp'].diff().dt.total_seconds() / 60  # in minutes
        df['time_since_last'].fillna(0, inplace=True)
        
        # Add context features
        for i in range(1, self.context_window + 1):
            df[f'prev_agent_{i}'] = df['agent'].shift(i)
            df[f'prev_action_{i}'] = df['action'].shift(i)
        
        # Fill NaN for context
        for col in df.columns:
            if 'prev_' in col:
                df[col].fillna('none', inplace=True)
        
        self.data_stats = {
            'total_records': len(df),
            'unique_agents': df['agent'].nunique(),
            'agent_list': df['agent'].unique().tolist(),
            'action_distribution': df['action'].value_counts().to_dict(),
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            }
        }
        
        return df
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train the probabilistic model and return all parameters."""
        
        logger.info("Training probabilistic model...")
        
        model_params = {
            'context_window': self.context_window,
            'data_stats': self.data_stats
        }
        
        # 1. Hourly activity patterns per agent
        logger.info("  Computing hourly patterns...")
        hourly_probs = {}
        for agent in df['agent'].unique():
            agent_df = df[df['agent'] == agent]
            hourly_counts = agent_df['hour'].value_counts(normalize=True).to_dict()
            # Ensure all hours are represented
            for hour in range(24):
                if hour not in hourly_counts:
                    hourly_counts[hour] = 0.001  # Small probability
            hourly_probs[agent] = hourly_counts
        model_params['hourly_probs'] = hourly_probs
        
        # 2. Day of week patterns
        logger.info("  Computing daily patterns...")
        daily_probs = {}
        for agent in df['agent'].unique():
            agent_df = df[df['agent'] == agent]
            daily_counts = agent_df['day_of_week'].value_counts(normalize=True).to_dict()
            # Ensure all days are represented
            for day in range(7):
                if day not in daily_counts:
                    daily_counts[day] = 0.1
            daily_probs[agent] = daily_counts
        model_params['daily_probs'] = daily_probs
        
        # 3. Agent transition matrix (who follows whom)
        logger.info("  Computing transition matrix...")
        transitions = defaultdict(lambda: defaultdict(int))
        for i in range(1, len(df)):
            prev_agent = df.iloc[i-1]['agent']
            curr_agent = df.iloc[i]['agent']
            transitions[prev_agent][curr_agent] += 1
        
        # Normalize transitions
        agent_transition_matrix = {}
        for prev_agent in transitions:
            total = sum(transitions[prev_agent].values())
            agent_transition_matrix[prev_agent] = {}
            for curr_agent in transitions[prev_agent]:
                agent_transition_matrix[prev_agent][curr_agent] = transitions[prev_agent][curr_agent] / total
        model_params['agent_transition_matrix'] = agent_transition_matrix
        
        # 4. Time gap distributions
        logger.info("  Computing time gap distributions...")
        time_gap_distributions = {}
        for agent in df['agent'].unique():
            agent_df = df[df['agent'] == agent]
            gaps = agent_df['time_since_last'][agent_df['time_since_last'] > 0].values
            if len(gaps) > 0:
                time_gap_distributions[agent] = {
                    'mean': float(np.mean(gaps)),
                    'std': float(np.std(gaps)),
                    'min': float(np.min(gaps)),
                    'max': float(np.max(gaps)),
                    'median': float(np.median(gaps)),
                    'percentile_25': float(np.percentile(gaps, 25)),
                    'percentile_75': float(np.percentile(gaps, 75))
                }
        model_params['time_gap_distributions'] = time_gap_distributions
        
        # 5. Reply probabilities
        logger.info("  Computing reply probabilities...")
        reply_probabilities = {}
        for agent in df['agent'].unique():
            agent_df = df[df['agent'] == agent]
            reply_count = (agent_df['action'] == 'reply').sum()
            total_count = len(agent_df)
            reply_probabilities[agent] = reply_count / total_count if total_count > 0 else 0
        model_params['reply_probabilities'] = reply_probabilities
        
        # 6. Context-based action probabilities
        logger.info("  Computing context patterns...")
        context_action_probs = {}
        for agent in df['agent'].unique():
            agent_df = df[df['agent'] == agent]
            context_patterns = defaultdict(lambda: {'post': 0, 'reply': 0})
            
            for _, row in agent_df.iterrows():
                # Use last 2 agents as context
                context = (row.get('prev_agent_1', 'none'), row.get('prev_agent_2', 'none'))
                action = row['action']
                context_patterns[context][action] += 1
            
            # Normalize
            normalized_patterns = {}
            for context, actions in context_patterns.items():
                total = sum(actions.values())
                if total > 0:
                    normalized_patterns[context] = {
                        'post': actions['post'] / total,
                        'reply': actions['reply'] / total
                    }
            context_action_probs[agent] = normalized_patterns
        model_params['context_action_probs'] = context_action_probs
        
        logger.info(f"✅ Model trained successfully!")
        logger.info(f"  - Agents: {list(agent_transition_matrix.keys())}")
        logger.info(f"  - Total interactions: {len(df)}")
        logger.info(f"  - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return model_params


def train_and_save_model(input_file: str, output_dir: str = "model_export"):
    """Main function to train and save the model."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize model
    model = AgentTimeProbabilisticModel(context_window=3)
    
    # Load and process data
    df = model.load_and_process_data(input_file)
    
    # Train model
    model_params = model.train(df)
    
    # Save model
    model_file = output_path / "probabilistic_model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model_params, f)
    
    logger.info(f"Model saved to {model_file}")
    logger.info(f"Model size: {model_file.stat().st_size / 1024:.2f} KB")
    
    # Save statistics
    stats_file = output_path / "model_statistics.json"
    stats = {
        'data_stats': model_params['data_stats'],
        'model_info': {
            'context_window': model_params['context_window'],
            'num_agents': len(model_params['agent_transition_matrix']),
            'agents': list(model_params['agent_transition_matrix'].keys()),
            'reply_rates': model_params['reply_probabilities'],
            'avg_time_gaps': {
                agent: gaps['mean'] 
                for agent, gaps in model_params['time_gap_distributions'].items()
            }
        },
        'training_date': datetime.now().isoformat()
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Statistics saved to {stats_file}")
    
    # Create documentation
    create_documentation(model_params, output_path)
    
    return model_params


def create_documentation(model_params: Dict, output_path: Path):
    """Create documentation for the model input/output format."""
    
    doc_content = """
# Probabilistic Model Input/Output Documentation

## Model Parameters Stored in PKL File

The `.pkl` file contains a dictionary with the following keys:

### 1. `agent_transition_matrix`
- **Format**: `Dict[str, Dict[str, float]]`
- **Example**: `{"AI_Agent": {"Crypto_Agent": 0.6, "AI_Agent": 0.4}}`
- **Usage**: Probability of agent B following agent A

### 2. `hourly_probs`
- **Format**: `Dict[str, Dict[int, float]]`
- **Example**: `{"AI_Agent": {0: 0.02, 1: 0.01, ..., 23: 0.05}}`
- **Usage**: Probability of agent posting at each hour (0-23)

### 3. `time_gap_distributions`
- **Format**: `Dict[str, Dict[str, float]]`
- **Contains**: mean, std, min, max, median, percentiles
- **Usage**: Time gaps between agent actions (in minutes)

### 4. `reply_probabilities`
- **Format**: `Dict[str, float]`
- **Example**: `{"AI_Agent": 0.3, "Crypto_Agent": 0.25}`
- **Usage**: Probability of agent replying vs posting

### 5. `context_action_probs`
- **Format**: `Dict[str, Dict[Tuple, Dict[str, float]]]`
- **Usage**: Action probabilities based on previous agents

## Input Format for Prediction

When using the trained model, you provide:

```python
# Current context (previous interactions)
context = [
    {
        'agent': 'AI_Agent',
        'action': 'post',
        'timestamp': datetime(2019, 5, 1, 10, 0, 0),
        'content': 'Previous content (optional)'
    },
    # ... more context items
]

# Current time for prediction
current_time = datetime(2019, 5, 1, 10, 30, 0)
```

## Output Format from Prediction

The model outputs:

```python
{
    'agent': 'Crypto_Agent',           # Predicted agent
    'action': 'reply',                  # Predicted action (post/reply)
    'timestamp': '2019-05-01 10:35:23', # Predicted time
    'probability': 0.73,                # Confidence score
    'time_gap_minutes': 5.38            # Time since last action
}
```

## Usage in LLaMA Pipeline

The output is formatted as:

```
"Crypto_Agent | Context: [AI_Agent +5m]: @AI Previous content | Action: reply"
```

This becomes the input to your fine-tuned LLaMA model to generate actual content.
"""
    
    doc_file = output_path / "MODEL_IO_FORMAT.md"
    with open(doc_file, 'w') as f:
        f.write(doc_content)
    
    # Add agent-specific stats
    agents_info = "\n\n## Agent Statistics\n\n"
    for agent in model_params['agent_transition_matrix'].keys():
        agents_info += f"### {agent}\n"
        agents_info += f"- Reply rate: {model_params['reply_probabilities'].get(agent, 0):.2%}\n"
        if agent in model_params['time_gap_distributions']:
            gaps = model_params['time_gap_distributions'][agent]
            agents_info += f"- Avg time between actions: {gaps['mean']:.1f} minutes\n"
            agents_info += f"- Median time: {gaps['median']:.1f} minutes\n"
        agents_info += "\n"
    
    with open(doc_file, 'a') as f:
        f.write(agents_info)
    
    logger.info(f"Documentation saved to {doc_file}")


if __name__ == "__main__":
    # Your specific file path
    input_file = "data/full_dataset_for_generation/enhanced_sorted_temporal_readable_88330_items_20250810.jsonl"
    output_dir = "model_export"
    
    logger.info("="*60)
    logger.info("Training Probabilistic Model for Your Data")
    logger.info("="*60)
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    
    # Train and save
    model_params = train_and_save_model(input_file, output_dir)
    
    print("\n" + "🎉"*20)
    print("SUCCESS! Model trained and ready to send to your colleague.")
    print(f"Send the file: {output_dir}/probabilistic_model.pkl")
    print("🎉"*20)