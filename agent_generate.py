import jax
import jax.numpy as jnp
from src.tools.executor import execute_code, extract_code_from_exec_tags
from src.memory.rag import rag_augmented_prompt

def agent_generate(
    state, 
    prompt, 
    char_to_idx, 
    idx_to_char, 
    rng, 
    max_iterations=5, # Hata düzeltme şansı sayısı
    max_tokens_per_iteration=300, 
    temperature=0.7,
    tool_start_token="<EXEC>",
    tool_end_token="</EXEC>",
    vector_store=None,
    verbose=False
):
    """
    Otonom Hata Düzeltme Döngüsü (Self-Correction Loop)
    """
    current_text = prompt
    
    # RAG Entegrasyonu (Opsiyonel)
    if vector_store and "<SEARCH>" in prompt:
        # ... (RAG kodları aynı kalabilir) ...
        pass

    for i in range(max_iterations):
        # 1. Metin Üretimi (Generation)
        input_ids = [char_to_idx.get(c, 0) for c in current_text]
        input_tensor = jnp.array([input_ids], dtype=jnp.int32)
        
        new_tokens = []
        curr_seq = input_tensor
        
        # Modelin cevabını bekle
        for _ in range(max_tokens_per_iteration):
            rng, step_rng = jax.random.split(rng)
            logits, _ = state.apply_fn({'params': state.params}, curr_seq, train=False)
            next_token = int(jax.random.categorical(step_rng, logits[0, -1, :] / temperature))
            
            new_tokens.append(next_token)
            curr_seq = jnp.concatenate([curr_seq, jnp.array([[next_token]])], axis=1)
            
            # </EXEC> veya durma sinyali gelirse kes
            decoded_chunk = "".join([idx_to_char.get(t, '') for t in new_tokens])
            if tool_end_token in decoded_chunk:
                break
        
        generated_chunk = "".join([idx_to_char.get(t, '') for t in new_tokens])
        current_text += generated_chunk
        
        # 2. Kod Yakalama ve Çalıştırma
        code = extract_code_from_exec_tags(generated_chunk, tool_start_token, tool_end_token)
        
        if code:
            if verbose: print(f"\n[DÖNGÜ {i+1}] Kod Çalıştırılıyor...")
            result = execute_code(code)
            
            # 3. KRİTİK NOKTA: Hata Analizi
            if "Error" in result or "Exception" in result:
                # Hata varsa, bunu modele "Sistem Mesajı" olarak geri besle
                # ve onu DÜŞÜNMEYE (THINK) zorla.
                feedback = f"\nSİSTEM: Kod hata verdi: {result.strip()}\nGÖREV: Hatayı analiz et, nedenini düşün ve kodu düzeltip tekrar yaz.\n"
                current_text += feedback
                if verbose: print(f"[HATA] Model uyarılıyor: {result.strip()}")
                
                # Döngü devam eder, model bir sonraki turda düzeltmeye çalışır.
            else:
                # Başarılıysa sonucu ekle
                current_text += f"\nSONUÇ: {result}\n"
                if verbose: print(f"[BAŞARI] Sonuç: {result.strip()}")
                # Başarılı işlemden sonra modelin son yorumunu yapmasına izin verip bitirebiliriz
                # veya devam ettirebiliriz. Genelde burada break yapılmaz, model "Tamamdır" diyene kadar devam eder.
        else:
            # Kod bloğu üretmediyse, cevap bitmiş demektir.
            break
            
    return current_text
