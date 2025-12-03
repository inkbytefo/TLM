# Agent Training Guide

Bu rehber, Spectral-JAX modelini bir problem-solving agent olarak eğitmek için gereken adımları açıklar.

## Görev Seti 6: Ajan Eğitimi ve Entegrasyon

### Genel Bakış

Model artık sadece metin üretmekle kalmıyor, aynı zamanda:
1. **Problem Anlama**: Verilen soruyu anlayabiliyor
2. **Kod Yazma**: Python kodu üretebiliyor (`<EXEC>` tagları arasında)
3. **Kod Çalıştırma**: Yazdığı kodu execute edebiliyor
4. **Sonuç Kullanma**: Sonucu alıp cevap üretebiliyor

## Adım 1: Agent Dataset Oluşturma

Dataset zaten oluşturuldu: `data/agent_dataset.txt`
- 1000 sentetik örnek
- Format: SORU → DÜŞÜNCE → EYLEM → SONUÇ → CEVAP
- Problem tipleri: Aritmetik, Faktöriyel, Liste işlemleri

Dataset'i yeniden oluşturmak için:
```bash
python -c "from src.data.agent_data import save_agent_dataset; save_agent_dataset('data/agent_dataset.txt', num_examples=1000, seed=42)"
```

## Adım 2: Agent Modelini Eğitme

### Konfigürasyon (config.py - AgentConfig)
```python
seq_len = 512           # Kısa sequence (agent örnekleri kısa)
batch_size = 16         # Büyük batch (sentetik veri)
num_steps = 3000        # Daha az adım (sentetik veri kolay)
learning_rate = 2e-4    # Daha yüksek LR
accum_steps = 2         # 16 * 2 = 32 efektif batch
temperature = 0.7       # Sampling sıcaklığı
```

### Eğitimi Başlatma
```bash
python run_agent_train.py
```

### Eğitim Süreci
- **50 adımda bir**: Training loss/accuracy
- **200 adımda bir**:
  - Validation metrics
  - Agent behavior samples (model `<EXEC>` taglerini kullanıyor mu?)
  - Checkpoint kaydetme
- **3000 adım sonunda**: Final test

### Beklenen Davranış

**İlk başta (0-500 adım):**
```
SORU: 345 * 982 nedir?
DÜŞÜNCE: Bu bir çarpma işlemi...
EYLEM: <random_chars>
```
Model henüz tool taglerini öğrenmedi.

**Ortada (500-1500 adım):**
```
SORU: 345 * 982 nedir?
DÜŞÜNCE: Bu bir çarpma işlemi...
EYLEM: <EXEC>prin(345*982</EXEC>
```
Tool taglerini öğrendi ama kod hatalı.

**Sonunda (1500-3000 adım):**
```
SORU: 345 * 982 nedir?
DÜŞÜNCE: Bu bir çarpma işlemi...
EYLEM: <EXEC>print(345 * 982)</EXEC>
SONUÇ: 338790
CEVAP: Sonuç 338790'dır.
```
✓ Model doğru çalışıyor!

## Adım 3: Agent'ı Test Etme

### Basit Test
```bash
python agent_generate.py
```
Bu, birkaç örnek prompt ile agent generation'ı test eder.

### Kapsamlı Test
```bash
python test_agent.py
```

Bu script 5 farklı problem tipi test eder:
1. **Basit Çarpma**: 12345 * 67890
2. **Faktöriyel**: 10!
3. **Liste Toplamı**: [10, 20, 30, 40, 50]
4. **Üs Alma**: 2^10
5. **Bölme**: 1000 // 7

### Test Sonuçları

Test sonuçları `agent_test_results.txt` dosyasına kaydedilir.

**Başarılı Test:**
```
✓ TEST PASSED - Found expected result: 838102050
```

**Başarısız Test:**
```
✗ TEST FAILED - Expected result not found
```

### Success Metrics
- **%100 başarı**: Agent tamamen çalışıyor 🎉
- **%60-99 başarı**: Kısmi başarı, daha fazla eğitim gerekebilir
- **<%60 başarı**: Daha fazla eğitim kesinlikle gerekli

## Adım 4: Agent'ı Kullanma

Eğitilen modeli kullanmak için:

```python
from agent_generate import agent_generate
from config import Config

config = Config()
# ... model yükleme kodu ...

result = agent_generate(
    state=state,
    prompt="SORU: 999 * 888 nedir?\nDÜŞÜNCE: ",
    char_to_idx=char_to_idx,
    idx_to_char=idx_to_char,
    rng=rng,
    max_iterations=5,
    temperature=0.7
)

print(result)
```

## Checkpoint Yönetimi

### Checkpointler Nerede?
```
checkpoints/
  agent_model/
    checkpoint_1000      # Son checkpoint
    checkpoint_2000
    best/
      checkpoint_XXXX    # En iyi model (en düşük val loss)
```

### Checkpoint Yükleme
```python
from flax.training import checkpoints

# Best model'i yükle
ckpt_dir = "checkpoints/agent_model/best"
state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=state)
```

## Troubleshooting

### Problem: Model tool taglerini öğrenmiyor
**Çözüm**: Daha fazla eğitim adımı
```python
# config.py
class AgentConfig:
    num_steps = 5000  # 3000'den artır
```

### Problem: Model kod yazamıyor
**Çözüm**: Learning rate'i artır
```python
class AgentConfig:
    learning_rate = 3e-4  # 2e-4'ten artır
```

### Problem: Test'ler fail oluyor
**Çözüm**:
1. Eğitim tamamlandı mı? (3000 adım)
2. Validation loss düştü mü? (~1.0 altında olmalı)
3. Training samples'da `<EXEC>` taglerini kullanıyor mu?

### Problem: GPU memory hatası
**Çözüm**: Batch size'ı küçült
```python
class AgentConfig:
    batch_size = 8  # 16'dan küçült
    accum_steps = 4  # 2'den artır (efektif batch aynı kalır)
```

## Sonraki Adımlar

Agent başarıyla çalıştığında:

1. **Daha Karmaşık Problemler**: Dataset'e daha zor problemler ekle
2. **Multi-Tool Support**: Birden fazla araç kullanımı
3. **Memory**: Agent'ın geçmiş işlemleri hatırlaması
4. **Planning**: Multi-step problem solving
5. **Self-Correction**: Hata yaptığında düzeltebilme

## Önemli Notlar

- Agent eğitimi Shakespeare'den daha hızlı (sentetik veri)
- 3000 adım yeterli olmalı
- Temperature 0.7 optimal (çok yüksek → rastgele, çok düşük → deterministic)
- Tool tagları (`<EXEC>`, `</EXEC>`) kritik - model bunları öğrenmeli

## Dosya Yapısı

```
TLM/
├── run_agent_train.py      # Agent eğitim scripti
├── test_agent.py            # Agent test scripti
├── agent_generate.py        # Agent generation loop
├── data/
│   └── agent_dataset.txt    # Sentetik agent dataset
├── src/
│   ├── data/
│   │   └── agent_data.py    # Dataset generator
│   └── tools/
│       └── executor.py      # Python code executor
├── checkpoints/
│   └── agent_model/         # Agent checkpoints
└── config.py                # AgentConfig
```

## Son Kontrol Listesi

- [ ] Dataset oluşturuldu (`data/agent_dataset.txt`)
- [ ] Agent eğitimi tamamlandı (`python run_agent_train.py`)
- [ ] Validation loss düştü (~1.0 altı)
- [ ] Model `<EXEC>` taglerini kullanıyor
- [ ] Test scripti çalıştırıldı (`python test_agent.py`)
- [ ] En az %60 test başarı oranı

Başarılar! 🚀
