# DPO Training Pipeline
This repository Servers SFT and RLHF pipelines for finetuning models 

## Steps to run the training pipeline
1. Configure accelerate
```bash
accelerate config
```

2. Run the following commands:
```bash
pip install -r requirements.txt
accelerate launch dpo_trainer.py
```
