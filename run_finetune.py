import os
import shutil
import shlex
from transformers import HfArgumentParser

from FlagEmbedding.abc.finetune.reranker import (
    AbsRerankerModelArguments,
    AbsRerankerDataArguments,
    AbsRerankerTrainingArguments
)
from FlagEmbedding.finetune.reranker.encoder_only.base import EncoderOnlyRerankerRunner


# TODO: make more argument configurable
# --fp16 seems not used and thus not taking effect.
def finetune(model_path, train_data_path, output_dir, batch_size=2, use_unsloth=True):
    args_str = """--model_name_or_path {} \
        --train_data {} \
        --cache_path ./cache/data \
        --train_group_size 8 \
        --query_max_len 512 \
        --passage_max_len 512 \
        --pad_to_multiple_of 8 \
        --knowledge_distillation True \
        --output_dir {} \
        --overwrite_output_dir \
        --learning_rate 3e-5 \
        --fp16 \
        --num_train_epochs 3 \
        --per_device_train_batch_size {} \
        --gradient_accumulation_steps 1 \
        --dataloader_drop_last True \
        --warmup_ratio 0.1 \
        --weight_decay 0.01 \
        --logging_steps 1 \
        --save_steps 1000 \
        --gradient_checkpointing
    """.format(model_path, train_data_path, output_dir, batch_size)
    args = shlex.split(args_str)
    parser = HfArgumentParser((AbsRerankerModelArguments, AbsRerankerDataArguments, AbsRerankerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)
    model_args: AbsRerankerModelArguments
    data_args: AbsRerankerDataArguments
    training_args: AbsRerankerTrainingArguments
    model_args.use_unsloth = use_unsloth

    runner = EncoderOnlyRerankerRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()

    # Remove checkpoint folder which includes optimizer states since no resume training is needed.
    checkpoint_dirs = [file for file in os.listdir(output_dir) if file.startswith("checkpoint-")]
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_dir = os.path.join(output_dir, checkpoint_dir)
        if os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            print(checkpoint_dir + " removed...")


finetune(model_path="/home/arda/kai/AI-NAS/bge-reranker-v2-m3",
         train_data_path="/home/arda/kai/AI-NAS/train_dataset_500.jsonl",
         output_dir="./bge-reranker-finetune",
         batch_size=4)
