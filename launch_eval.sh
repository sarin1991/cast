accelerate launch -m \
    lighteval accelerate \
    "pretrained=<path to model>" \
    "eval_tasks.txt" \
    --override-batch-size 32 \
    --output-dir="./evals/"