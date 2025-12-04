import jax
import jax.numpy as jnp
import numpy as np
from src.models.model import SpectralModel
from src.models.encoder import ByteLatentEncoder
from src.data.lra import tokenize
from config import Config

def verify_shapes():
    print("Verifying Shapes...")
    config = Config()
    
    # 1. Test Tokenizer
    text = "(MAX 2 3)"
    tokens = tokenize(text, config.data.seq_len)
    print(f"Tokenizer Output Shape: {tokens.shape}, Dtype: {tokens.dtype}")
    assert tokens.shape == (config.data.seq_len,)
    assert tokens.dtype == np.uint8
    print("Tokenizer Test Passed!")
    
    # 2. Test Encoder
    encoder = ByteLatentEncoder(hidden_dim=config.model.hidden_dim)
    dummy_input = jnp.zeros((1, config.data.seq_len), dtype=jnp.uint8)
    
    rng = jax.random.PRNGKey(0)
    params_enc = encoder.init(rng, dummy_input)
    encoded = encoder.apply(params_enc, dummy_input)
    
    print(f"Encoder Output Shape: {encoded.shape}")
    # Expected: (1, 2048/4, 256) = (1, 512, 256)
    expected_len = config.data.seq_len // 4
    assert encoded.shape == (1, expected_len, config.model.hidden_dim)
    print("Encoder Shape Test Passed!")
    
    # 3. Test Full Model
    model = SpectralModel(
        vocab_size=config.model.vocab_size,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_classes=10, # ListOps classes
        dropout_rate=config.model.dropout_rate
    )
    
    params_model = model.init(rng, dummy_input, train=False)
    output = model.apply(params_model, dummy_input, train=False)
    
    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (1, 10)
    print("Model Shape Test Passed!")

if __name__ == "__main__":
    try:
        verify_shapes()
        print("\nALL CHECKS PASSED SUCCESSFULLY.")
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
