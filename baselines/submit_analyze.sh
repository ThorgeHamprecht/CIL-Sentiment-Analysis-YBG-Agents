#!/bin/bash
# Submits error-analysis inference jobs for all trained models.
# Jobs queue automatically (cluster runs 1 at a time per cil account).
# Run this script directly on the cluster login node:
#   bash /home/thamprecht/cil/project/baselines/submit_analyze.sh

VENV="/work/scratch/thamprecht/cil/venv"
DATA_DIR="/work/scratch/thamprecht/cil/data"
SCRIPT="/home/thamprecht/cil/project/baselines/analyze.py"
LOG_DIR="/work/scratch/thamprecht/cil/logs"
BASE="/home/thamprecht/cil/project/baselines"

submit() {
    local model_type=$1
    local artifact_dir=$2
    local job_id
    job_id=$(sbatch \
        --job-name="analyze_${model_type}" \
        --time=00:30:00 \
        --account=cil \
        --output="${LOG_DIR}/analyze_${model_type}-%j.out" \
        --parsable \
        --wrap="
. /etc/profile.d/modules.sh
module add cuda/13.0
source ${VENV}/bin/activate
python ${SCRIPT} --model_type ${model_type} --artifact_dir ${artifact_dir} --data_dir ${DATA_DIR}
")
    echo "Submitted ${model_type} -> job ${job_id}  (log: ${LOG_DIR}/analyze_${model_type}-${job_id}.out)"
}

submit bilstm      "${BASE}/06_rnn_bilstm/artifacts"
submit transformer "${BASE}/07_transformer_custom/artifacts"
submit bilstm_attn "${BASE}/08_rnn_improved/artifacts"
submit mdeberta    "${BASE}/09_mdeberta_coral/artifacts"

echo ""
echo "All 4 jobs queued. Check status with: squeue -u thamprecht"
echo ""
echo "Once complete, scp results back with:"
echo "  scp thamprecht@<cluster>:${BASE}/06_rnn_bilstm/artifacts/val_preds.csv      ./val_preds_bilstm.csv"
echo "  scp thamprecht@<cluster>:${BASE}/07_transformer_custom/artifacts/val_preds.csv ./val_preds_transformer.csv"
echo "  scp thamprecht@<cluster>:${BASE}/08_rnn_improved/artifacts/val_preds.csv    ./val_preds_bilstm_attn.csv"
echo "  scp thamprecht@<cluster>:${BASE}/09_mdeberta_coral/artifacts/val_preds.csv  ./val_preds_mdeberta.csv"
