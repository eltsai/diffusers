Original readme moved to README_original.md

# Recipe for v7

**TPU Init**
```
export PROJECT_ID=tpu-prod-env-one-vm
export ZONE=us-central1-c
export TPU_NAME=elisatsai-vm
export ACCELERATOR_TYPE=tpu7x-16
export RUNTIME_VERSION=v2-alpha-tpu7-ubuntu2404

gcloud alpha compute tpus tpu-vm create ${TPU_NAME} --zone=${ZONE} --accelerator-type=${ACCELERATOR_TYPE} --project=${PROJECT_ID} --version=${RUNTIME_VERSION}
```

**1. Environment setup and install v7 specific libs**
```
setup_cmd="sudo apt update && \
sudo apt install -y python3.12-venv && \
python3 -m venv venv && \
source venv/bin/activate && \
rm -rf diffusers && \
git clone -b wan2-1-141s https://github.com/eltsai/diffusers.git || true && \
cd diffusers && \
git fetch origin && \
git reset --hard origin/wan2-1-141s

pip install -r requirements.txt
pip install -e . && \
sh -ex setup-dep.sh && \

gcloud storage cp gs://elisatsai/wan_2.1_setup_fix/autoencoder_kl_wan.py ~/venv/lib/python3.12/site-packages/maxdiffusion/models/wan/autoencoder_kl_wan.py && \
echo 'Patched maxdiffusion VAE library file.'"


gcloud compute tpus tpu-vm ssh --zone $ZONE $TPU_NAME --project $PROJECT_ID --worker=all --command="$setup_cmd"
```
and 
```
install_cmd="source ~/venv/bin/activate && \
WHEEL_NAME='libtpu-0.0.22.dev20250821+tpu7x-cp312-cp312-manylinux_2_31_x86_64.whl' && \
\
echo 'Installing JAX, JAXLIB, and libtpu dependencies (Step 1 of 3: Core Install)...' && \

pip install -U --pre jax jaxlib libtpu requests \
-i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
-f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
\
echo 'Installing Specific TPUv7x Driver (Step 2 of 3: Cleanup and Copy)...' && \

pip uninstall -y libtpu || true && \
gsutil cp gs://libtpu-tpu7x-releases/wheels/libtpu/\$WHEEL_NAME ~/\$WHEEL_NAME && \
\
echo 'Installing Specific TPUv7x Driver (Step 3 of 3: Final Install)...' && \

pip install --upgrade --no-deps ~/\$WHEEL_NAME && \
\
echo 'Verification complete.' && \
pip freeze | grep -E 'jax|jaxlib|libtpu'"

gcloud compute tpus tpu-vm ssh --zone $ZONE $TPU_NAME --project $PROJECT_ID --worker=all --command="$install_cmd"
```

**2. Run the model**
```
run_cmd="source ~/venv/bin/activate && \
killall -9 python || true && \
sleep 10 && \
export JAX_COMPILATION_CACHE_DIR="/dev/shm/jax_cache" && \
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1 && \
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 && \
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES='xla_gpu_per_fusion_autotune_cache_dir' && \
export TPU_NAME=\"$TPU_NAME\" && \
export HF_HUB_CACHE=/dev/shm/hf_cache && \
cd ~/diffusers && \
python wan_tx_splash_attn.py --profile"

gcloud compute tpus tpu-vm ssh --zone $ZONE $TPU_NAME --project $PROJECT_ID --worker=all --command="$run_cmd"
```