"""
Autonomous Agent Loop for Spectral-JAX (Hybrid Architecture).
"""

import time
import sys
import os
import jax
import jax.numpy as jnp
from flax.training import checkpoints

from config import Config, AgentLoopConfig
from src.models.gpt import SpectralGPT
from src.utils.common import setup_logger
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
        self.logger.info("Initializing Autonomous Agent Loop (Hybrid)...")
        
        # Initialize Model State
        self.rng = jax.random.PRNGKey(42)
        self.rng, init_rng = jax.random.split(self.rng)
        
        # Modeli oluştur (Trainer içindeki fonksiyonu kullanıyoruz)
        self.state = create_generative_train_state(init_rng, self.config)
        
        # Checkpoint Yükleme (Sırayla dener)
        ckpt_paths = [
            os.path.join(os.getcwd(), "checkpoints", "agent_model", "best"), # En iyi ajan
            os.path.join(os.getcwd(), "checkpoints", "phase1_hybrid_logic"), # Phase 1
        ]
        
        loaded = False
        for ckpt_dir in ckpt_paths:
            if os.path.exists(ckpt_dir):
                try:
                    # Hedef state ile checkpoint yapısı uyuşmalı
                    self.state = checkpoints.restore_checkpoint(ckpt_dir, target=self.state)
                    self.logger.info(f"✓ Model restored from: {ckpt_dir}")
                    loaded = True
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to load {ckpt_dir}: {e}")
        
        if not loaded:
            self.logger.warning("!!! NO CHECKPOINT FOUND. Starting with random weights !!!")

        # Bağlam ve Hafıza Başlatma
        self.context = []
        self.memory_state = None 
        
        print("\n[SİSTEM] Ajan hazır. Yazmaya başlayın (Çıkış için Ctrl+C)...")
        
    def step(self, token):
        """
        Tek bir token'ı işler, bağlamı günceller ve bir sonraki token'ı tahmin eder.
        """
        # 1. Bağlamı Güncelle
        self.context.append(token)
        
        # Sliding Window (Sonsuz hafıza için burası ileride memory_state ile birleşecek)
        max_len = self.config.model.seq_len
        if len(self.context) > max_len:
            self.context = self.context[-max_len:]
            
        # 2. Girdiyi Hazırla (Batch, SeqLen)
        input_ids = jnp.array([self.context], dtype=jnp.int32)
        
        # 3. İleri Geçiş
        self.rng, step_rng = jax.random.split(self.rng)
        
        # Hybrid model çağrısı
        # init_memory_state=None veriyoruz, çünkü her adımda tüm bağlamı (context)
        # tekrar işliyoruz (Hyena/Attention stateless çalışır, stateful inference optimizasyonu sonraki iş).
        logits, new_memory_states = self.state.apply_fn(
            {'params': self.state.params},
            input_ids,
            init_memory_state=None, 
            train=False,
            rngs={'dropout': step_rng}
        )
        
        # 4. Hafızayı Sakla (İleride stateful inference için)
        self.memory_state = new_memory_states
        
        # 5. Sampling (Sonraki Token)
        next_token_logits = logits[0, -1, :] / self.config.agent.temperature
        next_token = int(jax.random.categorical(step_rng, next_token_logits))
        
        return next_token

    def run(self):
        try:
            while True:
                try:
                    user_text = input("\nUSER: ")
                except EOFError:
                    break
                
                if not user_text:
                    continue

                # 1. Kullanıcı Girdisini İşle
                # UTF-8 encode edip byte byte veriyoruz
                input_tokens = list(user_text.encode('utf-8'))
                for token in input_tokens:
                    _ = self.step(token)
                
                # 2. Düşünme ve Konuşma Tetikleyicileri
                # Phase 1 modeli bunları henüz bilmez ama altyapı hazır olsun
                _ = self.step(THINK_TOKEN)
                last_output = self.step(SPEAK_TOKEN)
                
                print(f"[AGENT]: ", end='', flush=True)
                
                # 3. Üretim Döngüsü
                current_token = last_output
                for _ in range(500): # Maksimum cevap uzunluğu
                    
                    # Özel Token Kontrolleri
                    if current_token == WAIT_TOKEN:
                        break 
                    elif current_token == SILENCE_TOKEN:
                        break
                    elif current_token == THINK_TOKEN:
                        pass # Düşünmeye devam
                    elif current_token == SPEAK_TOKEN:
                        pass # Konuşmaya devam
                    else:
                        # Byte -> Karakter dönüşümü
                        if current_token < 256:
                            try:
                                char = bytes([current_token]).decode('utf-8')
                                print(char, end='', flush=True)
                            except:
                                pass # Yarım byte (multi-byte karakter) olabilir
                    
                    # Bir sonraki adımı tahmin et
                    current_token = self.step(current_token)
                    
                print() # Satır sonu

        except KeyboardInterrupt:
            print("\n[SİSTEM] Ajan kapatıldı.")

if __name__ == "__main__":
    agent = AgentLoop()
    agent.run()
