# Ubuntu-based container with ROCm PyTorch for training
# Uses AMD's official ROCm PyTorch image as base
FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0

# Set working directory
WORKDIR /app

# Install additional Python packages
RUN pip install --no-cache-dir \
    transformers==4.45.0 \
    datasets \
    tokenizers \
    accelerate \
    nltk \
    sentencepiece \
    scikit-learn \
    pandas \
    tqdm \
    mlflow \
    optuna

# Copy project files
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY data/ /app/data/
COPY scripts/ /app/scripts/

# Set environment variables for ROCm
ENV ROCR_VISIBLE_DEVICES=0
ENV HSA_OVERRIDE_GFX_VERSION=12.0.1
ENV HIP_FORCE_DEV_KERNARG=1

# Default command
CMD ["python", "scripts/test_text_models_training.py", "--model", "german_bert"]


