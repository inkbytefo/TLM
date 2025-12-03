
import jax.numpy as jnp
import numpy as np
from config import AgentLoopConfig

def get_autonomous_dataloader(batch_size=32, seq_len=128):
    """
    Generates synthetic data for the Autonomous Agent Loop.
    
    Data Format:
    - Mix of User Input, Silence, and Agent Responses.
    - Special Tokens: <SILENCE>, <WAIT>, <THINK>, <SPEAK>
    
    Scenarios:
    1. User speaks -> Agent listens -> Agent speaks
    2. Silence -> Agent waits
    3. Long Silence -> Agent thinks -> Agent speaks (Proactive)
    """
    
    SILENCE = AgentLoopConfig.SILENCE_TOKEN
    WAIT = AgentLoopConfig.WAIT_TOKEN
    THINK = AgentLoopConfig.THINK_TOKEN
    SPEAK = AgentLoopConfig.SPEAK_TOKEN
    
    vocab_range = (ord('a'), ord('z')) # Simple vocab for now
    
    while True:
        batch_inputs = []
        
        for _ in range(batch_size):
            # Choose a scenario
            scenario = np.random.choice(['reactive', 'waiting', 'proactive'], p=[0.5, 0.3, 0.2])
            
            seq = []
            
            if scenario == 'reactive':
                # User: "hello" -> Agent: <SPEAK> "hi"
                user_len = np.random.randint(3, 10)
                user_msg = np.random.randint(vocab_range[0], vocab_range[1], size=user_len).tolist()
                
                agent_len = np.random.randint(3, 10)
                agent_msg = np.random.randint(vocab_range[0], vocab_range[1], size=agent_len).tolist()
                
                seq.extend(user_msg)
                seq.append(SPEAK)
                seq.extend(agent_msg)
                
            elif scenario == 'waiting':
                # User: (Silence) -> Agent: <WAIT>
                silence_len = np.random.randint(5, 20)
                seq.extend([SILENCE] * silence_len)
                seq.append(WAIT)
                
            elif scenario == 'proactive':
                # User: (Long Silence) -> Agent: <THINK> ... <SPEAK> "hey"
                silence_len = np.random.randint(10, 30)
                seq.extend([SILENCE] * silence_len)
                seq.append(THINK)
                seq.extend([THINK] * np.random.randint(1, 5)) # Thinking steps
                seq.append(SPEAK)
                
                agent_len = np.random.randint(3, 10)
                agent_msg = np.random.randint(vocab_range[0], vocab_range[1], size=agent_len).tolist()
                seq.extend(agent_msg)
            
            # Pad or truncate
            if len(seq) > seq_len:
                seq = seq[:seq_len]
            else:
                seq = seq + [0] * (seq_len - len(seq)) # 0 as padding
                
            batch_inputs.append(seq)
            
        batch_inputs = np.array(batch_inputs, dtype=np.int32)
        
        # For generative training, input is x, target is x shifted by 1
        # But here we just yield the batch, the trainer handles shifting usually?
        # The curriculum script expects: {'input': ...}
        
        yield {'input': batch_inputs}

if __name__ == "__main__":
    # Test
    loader = get_autonomous_dataloader(batch_size=2, seq_len=20)
    batch = next(loader)
    print("Batch shape:", batch['input'].shape)
    print("Sample:", batch['input'][0])
