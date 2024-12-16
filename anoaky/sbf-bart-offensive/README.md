---
library_name: transformers
language:
- en
license: mit
base_model: facebook/bart-large-mnli
tags:
- generated_from_trainer
model-index:
- name: sbf-bart-offensive
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sbf-bart-offensive

This model is a fine-tuned version of [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) on the allenai/social-bias-frames dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 64
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 1

### Framework versions

- Transformers 4.47.0
- Pytorch 2.4.1
- Datasets 3.2.0
- Tokenizers 0.21.0
