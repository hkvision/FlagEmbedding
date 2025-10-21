### Prepare Unsloth Environment
```
conda create -n unsloth-xpu python=3.11 -y
conda activate unsloth-xpu
pip install torch==2.9.0 pytorch-triton-xpu==3.5.0 --index-url https://download.pytorch.org/whl/test

git clone -b on_device_finetuning_framework --single-branch https://github.com/intel-innersource/frameworks.ai.client-ai.unsloth-zoo-client.git
cd frameworks.ai.client-ai.unsloth-zoo-client
pip install -e .
cd ..

git clone -b on_device_finetuning_framework --single-branch https://github.com/intel-innersource/frameworks.ai.client-ai.unsloth-client.git
cd frameworks.ai.client-ai.unsloth-client
pip install -e .[huggingface]
cd ..

pip install "transformers<4.55" trl

# 装了unsloth之后重装triton
pip uninstall triton
pip uninstall pytorch-triton-xpu
pip install pytorch-triton-xpu==3.5.0 --index-url https://download.pytorch.org/whl/test

pip install bitsandbytes
# 跑qlora的话要自己build bitsandbytes
```

### FlagEmbedding
```
git clone https://github.com/hkvision/FlagEmbedding.git
应该装了上面的依赖之后FlagEmbedding需要的依赖都齐了，不需要安装FlagEmbedding，只要在FlagEmbedding目录下跑就好了
```


### Run
```
export UNSLOTH_DISABLE_STATISTICS=1
export USE_SYCL_KERNELS=0
export UNSLOTH_DISABLE_FAST_GENERATION=1

cd FlagEmbedding
python run_finetune.py

python __main__.py --model_name_or_path /home/arda/kai/AI-NAS/bge-reranker-v2-m3 \
    --train_data /home/arda/kai/AI-NAS/train_dataset_500.jsonl \
    --cache_path ./cache/data \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation True \
    --output_dir ./bge-reranker-finetune \
    --overwrite_output_dir \
    --learning_rate 6e-5 \
    --fp16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --logging_steps 1 \
    --save_steps 1000 \
    --gradient_checkpointing
```
