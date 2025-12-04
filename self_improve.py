import os
import jax
import time
from config import Config
from src.utils.common import setup_logger, set_seed
from src.training.trainer import create_generative_train_state
from flax.training import checkpoints
from src.data.text_generation import get_text_dataloaders
from agent_generate import agent_generate

# TensorFlow GPU engelleme
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

def main():
    logger = setup_logger()
    config = Config()
    set_seed(42)
    
    # Kayıt Dosyası
    output_file = "data/self_improved_data.txt"
    logger.info(f"Self-Improvement Modu Başlatılıyor... Hedef: {output_file}")
    
    # Modeli Yükle (En iyi checkpoint)
    # Phase 1 bitince burası 'checkpoints/phase1_hybrid_logic' olacak
    ckpt_dir = os.path.join("checkpoints", "phase1_hybrid_logic") 
    
    # Vocab yükle (Agent datasetinden)
    # Note: get_text_dataloaders might need adjustment if agent_dataset.txt doesn't exist or is different.
    # Assuming it exists or we can use a fallback.
    # If agent_dataset.txt is missing, we might need to point to another file or handle it.
    # For now, using the user provided code as is.
    if not os.path.exists('data/agent_dataset.txt'):
         logger.warning("data/agent_dataset.txt not found. Using data/turkish_academic.txt for vocab.")
         vocab_path = 'data/turkish_academic.txt'
    else:
         vocab_path = 'data/agent_dataset.txt'

    _, _, vocab_size, char_to_idx, idx_to_char = get_text_dataloaders(
        vocab_path, 512, 1
    )
    config.model.vocab_size = vocab_size
    
    rng = jax.random.PRNGKey(42)
    state = create_generative_train_state(rng, config)
    
    try:
        state = checkpoints.restore_checkpoint(ckpt_dir, target=state)
        logger.info("Model yüklendi.")
    except:
        logger.error("Model bulunamadı! Önce Phase 1 eğitimini bitirin.")
        return

    # Tohum Sorular (Seed Prompts)
    # Modeli tetiklemek için başlangıç noktaları
    seeds = [
        "SORU: Python ile 100'e kadar olan asal sayıları bulan bir kod yaz.\n",
        "SORU: Fibonacci dizisinin ilk 20 terimini hesapla.\n",
        "SORU: Bir listenin medyanını bulan fonksiyon yaz.\n",
        "SORU: Matris çarpımı yapan bir kod yaz.\n",
        "SORU: Verilen metindeki kelime sayısını hesapla.\n"
    ]
    
    total_generated = 0
    
    while True:
        for seed in seeds:
            logger.info(f"\n--- Yeni Görev: {seed.strip()} ---")
            
            rng, step_rng = jax.random.split(rng)
            
            # Ajanı çalıştır (Hata düzeltme modu açık)
            try:
                output = agent_generate(
                    state, seed, char_to_idx, idx_to_char, step_rng,
                    max_iterations=3, # 3 kere düzeltme hakkı ver
                    verbose=True
                )
                
                # Kalite Kontrolü (Quality Gate)
                # 1. Kod çalıştı mı? (SONUÇ var mı?)
                # 2. Hata mesajı son haliyle kaldı mı?
                if "SONUÇ:" in output and "Error" not in output.split("SONUÇ:")[-1]:
                    logger.info(">>> BAŞARILI! Veri setine ekleniyor.")
                    
                    # Veriyi kaydet
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(output + "\n\n<|endoftext|>\n\n")
                    
                    total_generated += 1
                else:
                    logger.warning(">>> BAŞARISIZ. Bu deneme atıldı.")
                    
            except Exception as e:
                logger.error(f"Kritik hata: {e}")
            
            if total_generated >= 1000:
                logger.info("Hedefe ulaşıldı (1000 yeni veri). Döngü duruyor.")
                return

if __name__ == "__main__":
    main()
