"""
Autonomous Agent Loop for Spectral-JAX.
Corrected to match model signature (init_memory_state).
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
        
        # Initialize Model State
        self.rng = jax.random.PRNGKey(42)
        self.rng, init_rng = jax.random.split(self.rng)
        self.state = create_generative_train_state(init_rng, self.config)
        
        # Load Checkpoint
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
            self.logger.error("!!! NO CHECKPOINT FOUND. AGENT IS UNTRAINED !!!")

        # Initialize Memory State
        self.memory_state = None 
        self.silence_counter = 0
        
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
                    
                    # 2. Inject THINK token (Matches training data: User -> THINK -> SPEAK)
                    _ = self.step(THINK_TOKEN)

                    # 3. Force "SPEAK" token to trigger response
                    # This acts as a "Go" signal for the model
                    last_output = self.step(SPEAK_TOKEN)
                    print(f"[AGENT]: ", end='', flush=True)
                    
                else:
                    # Silence
                    last_output = self.step(SILENCE_TOKEN)
                    self.silence_counter += 1
                
                # 3. Agent Reaction Loop
                current_token = last_output
                
                for _ in range(200): 
                    if current_token == WAIT_TOKEN:
                        if not user_text:
                            sys.stdout.write(".")
                            sys.stdout.flush()
                        break 
                    elif current_token == SILENCE_TOKEN:
                         break
                    elif current_token == THINK_TOKEN:
                        pass
                    elif current_token == SPEAK_TOKEN:
                        pass
                    else:
                        if current_token < 256:
                            try:
                                char = bytes([current_token]).decode('utf-8')
                                print(char, end='', flush=True)
                            except:
                                pass 
                    
                    current_token = self.step(current_token)
                    
                if user_text:
                    print() 

        except KeyboardInterrupt:
            print("\n[System] Agent Loop Terminated.")

if __name__ == "__main__":
    agent = AgentLoop()
    agent.run()
