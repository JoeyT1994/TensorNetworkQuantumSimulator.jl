"""Shared JAX runtime configuration for local experiments."""

import os

import jax


def configure_jax():
    """Configure FP64 without a persistent compilation cache."""
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)
    return None


JAX_COMPILATION_CACHE_DIR = configure_jax()
