import json
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# -----------------------------
# Load model and tokenizer
# -----------------------------
model_path = "./llama2-qlora-output-tech-RE"

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="./offload"
)

# -----------------------------
# Load real conversations
# -----------------------------
with open("my_conversations.json") as f:
    real_data = json.load(f)

# -----------------------------
# Helper: Extract timestamp from human message text
# -----------------------------
def extract_timestamp(human_msg):
    # Example human message:
    # "Generate a relevant tweet for posting on Friday afternoon at 2019-05-10 13:00:02..."
    # We extract the datetime string using simple parsing
    
    import re
    # Look for pattern like YYYY-MM-DD HH:MM:SS
    match = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", human_msg)
    if match:
        return datetime.strptime(match.group(), "%Y-%m-%d %H:%M:%S")
    else:
        # fallback if no timestamp found
        return datetime(2023, 1, 1)

# -----------------------------
# Build prompt function remains the same
# -----------------------------
def build_prompt(timestamp):
    day_of_week = timestamp.strftime("%A")
    time_of_day = timestamp.strftime("%p").lower()
    formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    return [
        {
            "from": "system",
            "value": "You are a AI_Agent, a specialized AI assistant focused on artificial intelligence and machine learning topics."
        },
        {
            "from": "human",
            "value": f"Generate a relevant tweet for posting on {day_of_week} {time_of_day} at {formatted_time}. "
                     "Consider the timing, day of week, and time of day when crafting your response to maximize engagement."
        }
    ]

# -----------------------------
# Generate conversations using real timestamps and real response length as max_new_tokens
# -----------------------------
new_conversations = []

for conv in real_data:  # limit to 10 for now
    # Extract human message and timestamp
    human_msg = next(msg for msg in conv["conversations"] if msg["from"] == "human")["value"]
    timestamp = extract_timestamp(human_msg)

    # Extract real assistant response length (in tokens)
    real_assistant_msg = next(msg for msg in conv["conversations"] if msg["from"] == "assistant")["value"]
    real_len = len(tokenizer.tokenize(real_assistant_msg))

    # Build prompt with extracted timestamp
    prompt = build_prompt(timestamp)
    input_text = "".join(f"{msg['from']}: {msg['value']}\n" for msg in prompt)

    # Tokenize input and generate with dynamic max_new_tokens
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=real_len, eos_token_id=tokenizer.eos_token_id)  # add buffer tokens
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant reply
    if "assistant:" in full_text:
        assistant_text = full_text.split("assistant:")[-1].strip()
    else:
        assistant_text = full_text.strip()

    # Append generated conversation
    new_conversations.append({
        "conversations": [
            prompt[0],  # system
            prompt[1],  # human
            {"from": "assistant", "value": assistant_text}
        ]
    })

# -----------------------------
# Save JSON output
# -----------------------------
with open("generated_conversations.json", "w") as f:
    json.dump(new_conversations, f, indent=2)

print("✅ Generated and saved 10 new conversations in generated_conversations.json")



