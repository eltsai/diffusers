killall -9 python || true && \
sleep 10 && \
export JAX_COMPILATION_CACHE_DIR="/dev/shm/jax_cache" && \
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1 && \
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 && \
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES='xla_gpu_per_fusion_autotune_cache_dir' && \
export TPU_NAME=\"$TPU_NAME\" && \
export HF_HUB_CACHE=/dev/shm/hf_cache && \
cd ~/diffusers && \
python wan_tx_splash_attn.py --profile --batch_size 1