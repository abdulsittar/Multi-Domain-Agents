
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


## Agent Statistics

### AI_Agent
- Reply rate: 8.87%
- Avg time between actions: 6.4 minutes
- Median time: 4.0 minutes

### Crypto_Agent
- Reply rate: 3.80%
- Avg time between actions: 6.3 minutes
- Median time: 4.0 minutes

