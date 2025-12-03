
import numpy as np
import random
from config import Config, AgentLoopConfig

class AutonomousDataGenerator:
    def __init__(self, batch_size, seq_len):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.config = Config()
        self.loop_conf = AgentLoopConfig # Use direct class reference if instance fails
        
        # Simple conversation pairs for training
        self.conversations = [
            ("Merhaba", "Merhaba, size nasıl yardımcı olabilirim?"),
            ("Nasılsın?", "Ben bir yapay zekayım, duygularım yok ama sistemlerim çalışıyor."),
            ("Python nedir?", "Python yüksek seviyeli bir programlama dilidir."),
            ("JAX nedir?", "JAX, Google tarafından geliştirilen yüksek performanslı bir kütüphanedir."),
            ("Test", "Sistem aktif ve dinliyor."),
            ("1+1 kaç?", "Sonuç 2'dir."),
            ("Bana bir şiir yaz", "Güller kırmızı, menekşeler mavi..."),
            ("Kod yaz", "print('Hello World')"),
        ]

    def encode(self, text):
        return [b for b in text.encode('utf-8')]

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
            # Decide scenario: 
            # 0: Pure Silence (User silent -> Agent waits)
            # 1: Interaction (User speaks -> Agent responds)
            scenario = random.random()
            
            seq = []
            
            if scenario < 0.3: # 30% Pure Silence
                # Input: SILENCE, Target: WAIT
                # We create a sequence where SILENCE is followed by WAIT
                # [SILENCE, WAIT, SILENCE, WAIT, ...]
                length = self.seq_len
                for _ in range(length // 2):
                    seq.append(SILENCE)
                    seq.append(WAIT)
                
            else: # 70% Conversation
                user_text, agent_text = random.choice(self.conversations)
                
                u_tokens = self.encode(user_text)
                a_tokens = self.encode(agent_text)
                
                # 1. User speaks
                seq.extend(u_tokens)
                
                # 2. Agent Thinks (Optional)
                if random.random() > 0.5:
                    seq.append(THINK)
                
                # 3. Agent Speaks
                seq.append(SPEAK)
                seq.extend(a_tokens)
                
                # 4. Return to Silence/Wait
                remaining = self.seq_len - len(seq)
                if remaining > 0:
                    for _ in range(remaining // 2):
                        seq.append(SILENCE)
                        seq.append(WAIT)
            
            # Truncate or Pad
            seq = seq[:self.seq_len]
            
            # Pad if too short
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

if __name__ == "__main__":
    # Test
    loader = get_autonomous_dataloader(batch_size=2, seq_len=20)
    batch = next(loader)
    print("Batch shape:", batch['input'].shape)
    print("Sample:", batch['input'][0])
