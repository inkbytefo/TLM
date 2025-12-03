"""
Autonomous Agent Loop for Spectral-JAX.
Corrected to load Curriculum Checkpoints and force interaction.
"""

import time
import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

from config import Config, AgentLoopConfig
from src.models.gpt import SpectralGPT
from src.utils.common import setup_logger, set_seed
from src.training.trainer import create_generative_train_state

# Special Tokens
SILENCE_TOKEN = AgentLoopConfig.SILENCE_TOKEN
WAIT_TOKEN = AgentLoopConfig.WAIT_TOKEN
THINK_TOKEN = AgentLoopConfig.THINK_TOKEN
SPEAK_TOKEN = AgentLoopConfig.SPEAK_TOKEN

class AgentLoop:
    def __init__(self):
        self.config = Config()
        self.logger = setup_logger()
        self.logger.info("Initializing Autonomous Agent Loop...")
        
        # Initialize Model State using Trainer Utility (Ensures consistency)
        self.rng = jax.random.PRNGKey(42)
        self.rng, init_rng = jax.random.split(self.rng)
        self.state = create_generative_train_state(init_rng, self.config)
        
        # --- CRITICAL: LOAD CURRICULUM CHECKPOINT ---
        # Priority order for checkpoints
        ckpt_paths = [
            os.path.join(os.getcwd(), "checkpoints", "curriculum", "Stage_4_Autonomy", "best"),
            os.path.join(os.getcwd(), "checkpoints", "autonomous_model"),
            os.path.join(os.getcwd(), "checkpoints", "agent_model", "best")
        ]
        
        loaded = False
        for ckpt_dir in ckpt_paths:
            if os.path.exists(ckpt_dir):
                try:
                    self.state = checkpoints.restore_checkpoint(ckpt_dir, target=self.state)
                    self.logger.info(f"✓ Model restored from: {ckpt_dir}")
                    loaded = True
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to load {ckpt_dir}: {e}")
        
        if not loaded:
            self.logger.error("!!! NO CHECKPOINT FOUND. AGENT IS UNTRAINED (RANDOM) !!!")
            self.logger.error("Please run: python train_curriculum.py")

        # Initialize Memory State
        self.memory_state = None 
        self.silence_counter = 0
        
        self.logger.info("Agent initialized and ready.")

    def step(self, input_token: int) -> int:
        """Run one step of the agent loop."""
        # Prepare input
        input_seq = jnp.array([[input_token]], dtype=jnp.int32) # (1, 1)
        
        self.rng, step_rng = jax.random.split(self.rng)
        
        # Forward pass
        logits, new_memory_state = self.state.apply_fn(
            {'params': self.state.params},
            input_seq,
            memory_state=self.memory_state,
            train=False
        )
        
        # Update memory state
        self.memory_state = new_memory_state
        
        # Sampling with Temperature
        next_token_logits = logits[0, -1, :] / self.config.agent.temperature
        next_token = int(jax.random.categorical(step_rng, next_token_logits))
        
        return next_token

    def run(self):
        print("\n=== Autonomous Agent Loop Started ===")
        print("Type something and press Enter. The agent is listening...")
        print("(Press Ctrl+C to exit)\n")
        
        try:
            while True:
                try:
                    user_text = input("\nUSER (or Enter for Silence): ")
                except EOFError:
                    break
                
                last_output = SILENCE_TOKEN

                if user_text:
                    # 1. Process User Input
                    input_tokens = list(user_text.encode('utf-8'))
                    for token in input_tokens:
                        _ = self.step(token)
                    
                    # 2. Force "SPEAK" token to trigger response
                    # This acts as a "Go" signal for the model
                    last_output = self.step(SPEAK_TOKEN)
                    print(f"[AGENT]: ", end='', flush=True)
                    
                else:
                    # Silence
                    last_output = self.step(SILENCE_TOKEN)
                    self.silence_counter += 1
                
                # 3. Agent Reaction Loop
                current_token = last_output
                
                for _ in range(200): # Max response length
                    # Handle Control Tokens
                    if current_token == WAIT_TOKEN:
                        if not user_text: # Only print wait if we are in silence mode
                            sys.stdout.write(".")
                            sys.stdout.flush()
                        break 
                        
                    elif current_token == SILENCE_TOKEN:
                         break
                         
                    elif current_token == THINK_TOKEN:
                        # Internal thought
                        pass
                        
                    elif current_token == SPEAK_TOKEN:
                        # Already handled or spontaneous speech
                        pass
                        
                    else:
                        # Text Output
                        if current_token < 256:
                            try:
                                char = bytes([current_token]).decode('utf-8')
                                print(char, end='', flush=True)
                            except:
                                pass 
                            
                    # Generate next token
                    current_token = self.step(current_token)
                    
                if user_text:
                    print() # Newline after response

        except KeyboardInterrupt:
            print("\n[System] Agent Loop Terminated.")

if __name__ == "__main__":
    agent = AgentLoop()
    agent.run()
