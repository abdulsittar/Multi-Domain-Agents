#!/usr/bin/env python3
"""
GPU Inference Script - Run this on the machine with GPU and fine-tuned LLaMA
=============================================================================
This script uses the pre-trained probabilistic model to generate conversations
with the fine-tuned LLaMA model.

Requirements:
- GPU with CUDA support
- Fine-tuned LLaMA model (LoRA weights)
- Probabilistic model file (.pkl) from colleague
- transformers, peft, torch libraries
"""

import json
import pickle
import torch
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import logging
from typing import Dict, List
import numpy as np
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class ConversationGenerator:
    """Generate conversations using probabilistic model + fine-tuned LLaMA."""
    
    def __init__(self, 
                 probabilistic_model_path: str,
                 llama_base_path: str = "meta-llama/Llama-2-7b-chat-hf",
                 llama_lora_path: str = "./llama2-lora-agent-simulation/final"):
        """
        Initialize with pre-trained probabilistic model and LLaMA.
        
        Args:
            probabilistic_model_path: Path to the .pkl file from colleague
            llama_base_path: Path to base LLaMA model
            llama_lora_path: Path to your fine-tuned LoRA weights
        """
        
        # Load probabilistic model
        logger.info(f"Loading probabilistic model from {probabilistic_model_path}...")
        with open(probabilistic_model_path, 'rb') as f:
            self.prob_model = pickle.load(f)
        
        logger.info(f"✅ Probabilistic model loaded successfully")
        logger.info(f"  - Agents: {list(self.prob_model['agent_transition_matrix'].keys())}")
        logger.info(f"  - Context window: {self.prob_model.get('context_window', 3)}")
        
        # Load LLaMA model
        logger.info("Loading LLaMA model and tokenizer...")
        self.tokenizer = LlamaTokenizer.from_pretrained(llama_base_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check for GPU
        if torch.cuda.is_available():
            logger.info(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
            device_map = "auto"
        else:
            logger.warning("⚠️ No GPU detected, using CPU (will be slow)")
            device_map = "cpu"
        
        # Load base model
        base_model = LlamaForCausalLM.from_pretrained(
            llama_base_path,
            device_map=device_map,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            load_in_4bit=True if torch.cuda.is_available() else False
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, llama_lora_path)
        self.model.eval()
        logger.info("✅ LLaMA model loaded with LoRA weights")
        
        # Initialize conversation storage
        self.conversations = []
    
    def predict_next_agent(self, context: List[Dict], current_time: datetime) -> tuple:
        """Predict which agent acts next using the probabilistic model."""
        
        transitions = self.prob_model['agent_transition_matrix']
        hourly_probs = self.prob_model['hourly_probs']
        
        # Get available agents
        agents = list(transitions.keys())
        if not agents:
            return "AI_Agent", 0.5
        
        # Calculate probabilities
        agent_probs = {}
        
        for agent in agents:
            prob = 0.0
            
            # Transition probability (50% weight)
            if context and context[-1]['agent'] in transitions:
                trans_prob = transitions[context[-1]['agent']].get(agent, 0.1)
            else:
                trans_prob = 1.0 / len(agents)
            prob += 0.5 * trans_prob
            
            # Hourly activity (50% weight)
            hour = current_time.hour
            hour_prob = hourly_probs.get(agent, {}).get(hour, 0.05)
            prob += 0.5 * hour_prob
            
            agent_probs[agent] = prob
        
        # Normalize
        total = sum(agent_probs.values())
        if total > 0:
            for agent in agent_probs:
                agent_probs[agent] /= total
        
        # Sample agent
        agents_list = list(agent_probs.keys())
        probs_list = list(agent_probs.values())
        selected = np.random.choice(agents_list, p=probs_list)
        
        return selected, agent_probs[selected]
    
    def predict_action_and_time(self, agent: str, last_time: datetime) -> tuple:
        """Predict action type and timing."""
        
        # Get time gap distribution
        time_gaps = self.prob_model.get('time_gap_distributions', {})
        
        if agent in time_gaps:
            dist = time_gaps[agent]
            mean = dist['mean']
            std = dist['std']
            # Sample from log-normal
            gap = np.exp(np.random.normal(np.log(mean + 1), np.log(std + 1) * 0.5))
            gap = np.clip(gap, dist.get('min', 1), dist.get('max', 120))
        else:
            gap = np.random.exponential(30)
        
        next_time = last_time + timedelta(minutes=gap)
        
        # Predict action
        reply_probs = self.prob_model.get('reply_probabilities', {})
        reply_prob = reply_probs.get(agent, 0.3)
        action = 'reply' if np.random.random() < reply_prob else 'post'
        
        return action, next_time
    
    def format_llm_input(self, agent: str, action: str, context: List[Dict]) -> str:
        """Format input for fine-tuned model."""
        
        if not context:
            return f"Instruction: {agent} | Context:  | Action: {action}\nOutput:"
        
        # Build context matching training format
        context_parts = []
        base_time = context[-1]['timestamp']
        
        for ctx in context:
            time_diff = int((base_time - ctx['timestamp']).total_seconds() / 60)
            content = ctx.get('content', '@' + ctx['agent'].split('_')[0] + ' [content]')
            if len(content) > 80:
                content = content[:80] + "..."
            ctx_str = f"[{ctx['agent']} +{abs(time_diff)}m]: {content}"
            context_parts.append(ctx_str)
        
        context_str = " | ".join(context_parts)
        return f"Instruction: {agent} | Context: {context_str} | Action: {action}\nOutput:"
    
    def generate_content(self, prompt: str, **kwargs) -> str:
        """Generate content using fine-tuned LLaMA."""
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_new_tokens', 150),
                temperature=kwargs.get('temperature', 0.8),
                top_p=kwargs.get('top_p', 0.9),
                do_sample=kwargs.get('do_sample', True),
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract output
        if "Output:" in generated:
            content = generated.split("Output:")[-1].strip()
        else:
            content = generated[len(prompt):].strip()
        
        return content
    
    def generate_conversation(self, 
                            start_time: datetime = None,
                            num_interactions: int = 10,
                            verbose: bool = True) -> List[Dict]:
        """Generate a complete conversation."""
        
        if start_time is None:
            start_time = datetime.now()
        
        conversation = []
        current_time = start_time
        context_window = self.prob_model.get('context_window', 3)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating conversation with {num_interactions} interactions")
        logger.info(f"{'='*60}")
        
        for i in range(num_interactions):
            # Get context
            context = conversation[-context_window:] if len(conversation) >= context_window else conversation
            
            # Predict agent
            agent, prob = self.predict_next_agent(context, current_time)
            
            # Predict action and time
            action, next_time = self.predict_action_and_time(agent, current_time)
            
            if verbose:
                logger.info(f"\n[{i+1}/{num_interactions}] {agent} - {action}")
                logger.info(f"  Time: {next_time.strftime('%H:%M:%S')} (prob: {prob:.3f})")
            
            # Generate content
            llm_input = self.format_llm_input(agent, action, context)
            
            start_gen = time.time()
            content = self.generate_content(llm_input)
            gen_time = time.time() - start_gen
            
            if verbose:
                logger.info(f"  Generated in {gen_time:.2f}s: {content[:100]}...")
            
            # Store interaction
            interaction = {
                'agent': agent,
                'action': action,
                'timestamp': next_time,
                'content': content,
                'probability': prob,
                'generation_time': gen_time
            }
            
            conversation.append(interaction)
            current_time = next_time
        
        self.conversations.append(conversation)
        return conversation
    
    def save_conversations(self, output_dir: str = "generated_conversations"):
        """Save all generated conversations."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save conversations
        all_convs = []
        for conv_idx, conv in enumerate(self.conversations):
            conv_data = []
            for interaction in conv:
                item = {
                    'agent': interaction['agent'],
                    'action': interaction['action'],
                    'timestamp': interaction['timestamp'].strftime("%A %B %d %Y, %H:%M:%S"),
                    'content': interaction['content'],
                    'probability': interaction['probability']
                }
                conv_data.append(item)
            all_convs.append(conv_data)
        
        # Save as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conv_file = output_path / f"conversations_{timestamp}.json"
        with open(conv_file, 'w') as f:
            json.dump(all_convs, f, indent=2)
        
        # Save as readable text
        text_file = output_path / f"conversations_{timestamp}.txt"
        with open(text_file, 'w') as f:
            for conv_idx, conv in enumerate(all_convs):
                f.write(f"\n{'='*60}\n")
                f.write(f"CONVERSATION {conv_idx + 1}\n")
                f.write(f"{'='*60}\n\n")
                
                for item in conv:
                    f.write(f"[{item['timestamp'].split(',')[1].strip()}] ")
                    f.write(f"{item['agent']} ({item['action']})\n")
                    f.write(f"{item['content']}\n\n")
        
        logger.info(f"\n✅ Conversations saved to:")
        logger.info(f"  - JSON: {conv_file}")
        logger.info(f"  - Text: {text_file}")
        
        return str(conv_file)


def main():
    parser = argparse.ArgumentParser(description='Generate conversations using probabilistic model + LLaMA')
    parser.add_argument('--model', type=str, required=True, help='Path to probabilistic model .pkl file')
    parser.add_argument('--llama-base', type=str, default='meta-llama/Llama-2-7b-chat-hf', 
                       help='Path to base LLaMA model')
    parser.add_argument('--llama-lora', type=str, default='./llama2-lora-agent-simulation/final',
                       help='Path to LoRA weights')
    parser.add_argument('--num-conversations', type=int, default=3, help='Number of conversations to generate')
    parser.add_argument('--interactions', type=int, default=10, help='Interactions per conversation')
    parser.add_argument('--output', type=str, default='generated_conversations', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = ConversationGenerator(
        probabilistic_model_path=args.model,
        llama_base_path=args.llama_base,
        llama_lora_path=args.llama_lora
    )
    
    # Generate conversations
    total_time = 0
    for i in range(args.num_conversations):
        logger.info(f"\n🚀 Generating conversation {i+1}/{args.num_conversations}")
        
        start = time.time()
        conversation = generator.generate_conversation(
            start_time=datetime.now() + timedelta(days=i),
            num_interactions=args.interactions,
            verbose=True
        )
        elapsed = time.time() - start
        total_time += elapsed
        
        logger.info(f"\n⏱️ Conversation generated in {elapsed:.2f} seconds")
    
    # Save results
    generator.save_conversations(args.output)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("GENERATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total conversations: {args.num_conversations}")
    logger.info(f"Total interactions: {args.num_conversations * args.interactions}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Average per interaction: {total_time/(args.num_conversations * args.interactions):.2f} seconds")
    
    if torch.cuda.is_available():
        logger.info(f"GPU memory used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # Demo mode
        print("\n" + "="*60)
        print("DEMO MODE - Using example paths")
        print("="*60)
        print("\nUsage: python run_simulation_gpu.py --model model.pkl --llama-lora ./fine-tuned")
        print("\nFor demo, assuming:")
        print("  - Probabilistic model: probabilistic_model.pkl")
        print("  - LLaMA LoRA: ./llama2-lora-agent-simulation/final")
        
        # Check if model exists
        if Path("probabilistic_model.pkl").exists():
            generator = ConversationGenerator(
                probabilistic_model_path="probabilistic_model.pkl",
                llama_base_path="meta-llama/Llama-2-7b-chat-hf",
                llama_lora_path="./llama2-lora-agent-simulation/final"
            )
            
            conversation = generator.generate_conversation(
                num_interactions=5,
                verbose=True
            )
            
            generator.save_conversations()
        else:
            print("\n❌ No probabilistic model found. Get the .pkl file from your colleague first!")
    else:
        main()