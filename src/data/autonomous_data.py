import numpy as np
import random
from config import Config, AgentLoopConfig

class AutonomousDataGenerator:
    def __init__(self, batch_size, seq_len):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.config = Config()
        self.loop_conf = AgentLoopConfig
        
        # Templates for dynamic generation
        self.greetings = ["Merhaba", "Selam", "Hey", "Nasılsın", "Günaydın"]
        self.responses = ["Merhaba!", "Selam, dinliyorum.", "İyiyim, sen nasılsın?", "Buyrun?"]
        
    def encode(self, text):
        return [b for b in text.encode('utf-8')]

    def generate_math_pair(self):
        """Generates a random math problem and answer."""
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        op = random.choice(['+', '-', '*'])
        
        if op == '+': res = a + b
        elif op == '-': res = a - b
        elif op == '*': res = a * b
        
        questions = [
            f"{a}{op}{b}",
            f"{a} {op} {b} kaç?",
            f"{a} artı {b}",
            f"Hesapla: {a}{op}{b}"
        ]
        answers = [
            f"{res}",
            f"Sonuç {res}",
            f"{res} eder.",
            f"Cevap: {res}"
        ]
        return random.choice(questions), random.choice(answers)

    def generate_batch(self):
        """
        Generates a batch of sequences simulating autonomous interaction.
        Format: [User Input] -> [Response] -> [Silence/Wait]
        """
        batch_inputs = []
        
        SILENCE = self.loop_conf.SILENCE_TOKEN
        WAIT = self.loop_conf.WAIT_TOKEN
        THINK = self.loop_conf.THINK_TOKEN
        SPEAK = self.loop_conf.SPEAK_TOKEN

        for _ in range(self.batch_size):
            scenario = random.random()
            seq = []
            
            if scenario < 0.2: # 20% Pure Silence
                length = self.seq_len
                for _ in range(length // 2):
                    seq.append(SILENCE)
                    seq.append(WAIT)
                
            else: # 80% Interaction
                # Decide type of interaction
                if random.random() < 0.5:
                    # Math/Logic (Dynamic)
                    user_text, agent_text = self.generate_math_pair()
                else:
                    # Chat (Semi-dynamic)
                    user_text = random.choice(self.greetings)
                    agent_text = random.choice(self.responses)
                
                u_tokens = self.encode(user_text)
                a_tokens = self.encode(agent_text)
                
                # 1. User speaks
                seq.extend(u_tokens)
                
                # 2. Agent Thinks (Optional but good for structure)
                # Always add THINK for consistency in this training phase
                seq.append(THINK)
                
                # 3. Agent Speaks
                seq.append(SPEAK)
                seq.extend(a_tokens)
                
                # 4. Return to Silence/Wait
                remaining = self.seq_len - len(seq)
                if remaining > 0:
                    # Fill rest with silence/wait pairs
                    for _ in range(remaining // 2 + 1):
                        if len(seq) < self.seq_len:
                            seq.append(SILENCE)
                        if len(seq) < self.seq_len:
                            seq.append(WAIT)
            
            # Truncate or Pad
            seq = seq[:self.seq_len]
            
            if len(seq) < self.seq_len:
                pad = [0] * (self.seq_len - len(seq))
                seq.extend(pad)
                
            batch_inputs.append(seq)

        return {
            'input': np.array(batch_inputs, dtype=np.int32)
        }

def get_autonomous_dataloader(batch_size, seq_len):
    gen = AutonomousDataGenerator(batch_size, seq_len)
    while True:
        yield gen.generate_batch()
