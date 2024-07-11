#!/bin/bash
set -e
set -x

FRAC=0
OUTDIR=out/data

# Request user input if not provided via command-line arguments
if [ -z "$GPUS" ]; then
    read -p "Enter the number of GPUs: " GPUS
fi

if [ -z "$MODEL" ]; then
    read -p "Enter the model path: " MODEL
fi

if [ -z "$MAX_LEN" ]; then
    read -p "Enter the max length: " MAX_LEN
fi

if [ -z "$FRAC_LEN" ]; then
    read -p "Enter the fraction length: " FRAC_LEN
fi

if [ -z "$PAIRS" ]; then
    read -p "Enter the number of pairs: " PAIRS
fi

if [ -z "$PROMPTS" ]; then
    read -p "Enter the prompts file path: " PROMPTS
fi

# Parsing command-line arguments
while [ "$#" -gt 0 ]; do
    case $1 in
    --GPUS)
        GPUS="$2"
        shift
        ;;
    --MODEL)
        MODEL="$2"
        shift
        ;;
    --OUTDIR)
        OUTDIR="$2"
        shift
        ;;
    --MAX_LEN)
        MAX_LEN="$2"
        shift
        ;;
    --FRAC_LEN)
        FRAC_LEN="$2"
        shift
        ;;
    --PAIRS)
        PAIRS="$2"
        shift
        ;;
    --FRAC)
        FRAC="$2"
        shift
        ;;
    --PROMPTS)
        PROMPTS="$2"
        shift
        ;;
    *)
        echo "Unknown parameter passed: $1"
        exit 1
        ;;
    esac
    shift
done

# GPU setup
CUDA_VISIBLE_DEVICES=""
for ((i=0; i<GPUS; i++)); do
    if [ $i -ne 0 ]; then
        CUDA_VISIBLE_DEVICES+=","
    fi
    CUDA_VISIBLE_DEVICES+="$i"
done
export CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

#####################
# Generate Data
#####################

(
    for gpu_id in $(seq 0 $((GPUS-1))); do
        CUDA_VISIBLE_DEVICES=$gpu_id python3 generate.py \
            --model "$MODEL" \
            --maxlen $MAX_LEN \
            --output_dir "$OUTDIR" \
            --prompts "$PROMPTS" \
            --pairs "$PAIRS" \
            --world_size 1 \
            --frac_len $FRAC_LEN \
            --data_frac $gpu_id > "out/logs/output_log_${gpu_id}.txt" 2>&1 &
    done
    wait
) &
all_gen=$!

wait $all_gen

python3 combine_generate.py --output_dir "$OUTDIR" --numgpu $GPUS --pairs $PAIRS

#####################
# Rank Data
#####################

(
    for gpu_id in $(seq 0 $((GPUS-1))); do
        CUDA_VISIBLE_DEVICES=$gpu_id python3 rank.py \
            --model "$MODEL" \
            --output_dir "$OUTDIR" \
            --pairs "$PAIRS" \
            --numgpu $GPUS \
            --frac_len $FRAC_LEN \
            --data_frac $gpu_id \
            --gpu $gpu_id \
            --prompts "$PROMPTS" > "out/logs/rank_log_${gpu_id}.txt" 2>&1 &
    done
    wait
) &
all_rank=$!

wait $all_rank

python3 compute_prob.py --output_dir $OUTDIR --pairs $PAIRS --frac_len $FRAC_LEN --prompts $PROMPTS
