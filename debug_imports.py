try:
    import jax
    import flax
    import optax
    import hydra
    print("Imports successful")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
