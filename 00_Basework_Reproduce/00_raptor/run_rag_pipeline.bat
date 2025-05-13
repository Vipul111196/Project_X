#!/bin/bash
#SBATCH --job-name=rag-pipeline
#SBATCH --partition=A100-40GB,RTXA6000,A100-80GB,H100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rag_pipeline_%j.out
#SBATCH --error=logs/rag_pipeline_%j.err

echo "Running on $(hostname) at $(date)"

srun \
  --container-image=/netscratch/tyagi/ollama.sqsh \
  --container-mounts=/netscratch/tyagi/ollama_configs:/root/.ollama,/netscratch/tyagi/my_python_env:/root/env,/netscratch/tyagi/My_Work/Code/raptor/secrets:/root/secrets,/netscratch/tyagi:/netscratch/tyagi,/netscratch/tyagi/hf_models:/root/.cache/huggingface \
  --container-workdir="/netscratch/tyagi/My_Work/Code/raptor" \
  bash -c "
    source /root/env/miniconda/etc/profile.d/conda.sh && \
    conda activate raptor && \
    python tree_generator.py
  "

echo "Finished at $(date)"
