import numpy as np
import random
import os
from config import Config, AgentLoopConfig

class AutonomousDataGenerator:
    def __init__(self, batch_size, seq_len):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.config = Config()
        self.loop_conf = AgentLoopConfig
        
        # Load Shakespeare for language capability
        self.text_data = ""
        try:
            with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
                self.text_data = f.read()
        except:
            self.text_data = "Python is a programming language. JAX is a library. The sky is blue." * 100

        # Knowledge Base (General Culture)
        self.qa_pairs = [
            ("Nasılsın?", "Sistemlerim nominal çalışıyor, teşekkürler."),
            ("Sen kimsin?", "Ben Spectral-JAX mimarisiyle çalışan otonom bir yapay zekayım."),
            ("Görevin ne?", "Kullanıcı komutlarını dinlemek, kod çalıştırmak ve yardımcı olmak."),
            ("Python nedir?", "Python, okunabilirliği yüksek, yorumlanan bir programlama dilidir."),
            ("JAX nedir?", "JAX, Google tarafından geliştirilen, GPU/TPU hızlandırmalı bir sayısal hesaplama kütüphanesidir."),
            ("Evrenin anlamı ne?", "42."),
            ("Hangi yıldayız?", "Veri tabanım güncel değil ama 2025 yılındayız diyebilirim."),
        ]

    def encode(self, text):
        return [b for b in text.encode('utf-8', errors='ignore')]

    def get_random_text_segment(self, length=50):
        if len(self.text_data) < length: return "Text data too short."
        start = random.randint(0, len(self.text_data) - length - 1)
        return self.text_data[start:start+length]

    def generate_math(self):
        a = random.randint(1, 999)
        b = random.randint(1, 999)
        op = random.choice(['+', '-', '*'])
        if op == '+': res = a + b
        elif op == '-': res = a - b
        elif op == '*': res = a * b
        return f"{a} {op} {b}", str(res)

    def generate_batch(self):
        batch_inputs = []
        
        SILENCE = self.loop_conf.SILENCE_TOKEN
        WAIT = self.loop_conf.WAIT_TOKEN
        THINK = self.loop_conf.THINK_TOKEN
        SPEAK = self.loop_conf.SPEAK_TOKEN

        for _ in range(self.batch_size):
            scenario = random.random()
            seq = []
            
            # --- SCENARIO 1: SILENCE (20%) ---
            if scenario < 0.2:
                for _ in range(self.seq_len // 2):
                    seq.append(SILENCE)
                    seq.append(WAIT)
            
            # --- SCENARIO 2: MATH (30%) ---
            elif scenario < 0.5:
                q, a = self.generate_math()
                seq.extend(self.encode(q))
                seq.append(THINK) # Düşün
                seq.append(SPEAK) # Konuş
                seq.extend(self.encode(a))
                
            # --- SCENARIO 3: CHAT / QA (30%) ---
            elif scenario < 0.8:
                q, a = random.choice(self.qa_pairs)
                seq.extend(self.encode(q))
                seq.append(THINK)
                seq.append(SPEAK)
                seq.extend(self.encode(a))
                
            # --- SCENARIO 4: LITERATURE / READING (20%) ---
            else:
                # User asks to read/complete text
                text_segment = self.get_random_text_segment(100)
                prompt = "Oku: "
                seq.extend(self.encode(prompt))
                seq.append(THINK)
                seq.append(SPEAK)
                seq.extend(self.encode(text_segment))

            # Fill rest with Silence/Wait to teach returning to idle
            remaining = self.seq_len - len(seq)
            if remaining > 0:
                for _ in range(remaining // 2 + 1):
                    if len(seq) < self.seq_len: seq.append(SILENCE)
                    if len(seq) < self.seq_len: seq.append(WAIT)
            
            # Truncate/Pad
            seq = seq[:self.seq_len]
            if len(seq) < self.seq_len:
                seq.extend([0] * (self.seq_len - len(seq)))
                
            batch_inputs.append(seq)

        return {'input': np.array(batch_inputs, dtype=np.int32)}

def get_autonomous_dataloader(batch_size, seq_len):
    gen = AutonomousDataGenerator(batch_size, seq_len)
    while True:
        yield gen.generate_batch()
