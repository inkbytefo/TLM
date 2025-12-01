import logging
import random
import numpy as np
import jax

def setup_logger(name: str = "spectral_jax", level: int = logging.INFO) -> logging.Logger:
    """Sets up a logger with standard formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def set_seed(seed: int):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # JAX PRNG keys are handled explicitly, but we can set a global config if needed
    # jax.config.update("jax_enable_x64", True) # Example config
