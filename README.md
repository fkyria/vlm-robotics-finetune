# Fine-Tuning a Vision-Language Model on Robot Manipulation Data

QLoRA fine-tuning of [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
on [Robo2VLM-1](https://huggingface.co/datasets/keplerccc/Robo2VLM-1), a multiple-choice
visual question-answering dataset grounded in real robot manipulation footage.

## What this does

Given an image from a robot's manipulation task and a multiple-choice question about it
(e.g. grasp stability, task success, object/goal configuration), the model is trained to
output the correct answer letter.

## Method

- **Base model:** Qwen2-VL-2B-Instruct, loaded in 4-bit (QLoRA / NF4 quantization) to fit
  on a 16GB T4.
- **Adaptation:** LoRA (`r=16, alpha=32`) applied to all attention and MLP projection
  layers in the language model backbone. The vision encoder is left frozen.
- **Data:** 200 training / 20 validation examples from Robo2VLM-1, streamed
  rather than downloaded in bulk (the full dataset is 107GB).
- **Training:** effective batch size 8 (via gradient accumulation), 3 epochs, with
  step-level evaluation, best-checkpoint selection on eval loss, and early stopping.


## Results

| Metric | Value |
|---|---|
| Training examples | 200 |
| Validation examples | 20 |
| Test examples | 100 |
| Final training loss | 0.244 |
| Final evaluation loss | 0.358 (step 75) |
| Validation accuracy | 80.0% (16/20) |
| **Test accuracy** | **67.0% (67/100)** |    


**Notes:**
- The best checkpoint (lowest validation loss) was the very last one, at step 75, meaning
validation loss was still improving right up to the end of the 3-epoch run rather than
having plateaued or started rising. This suggests the model likely hadn't finished
learning yet, and more epochs (or more training data) would plausibly improve results
further rather than risk overfitting.

- As expected, validation accuracy is higher than test accuracy → the validation set indirectly influenced which checkpoint got kept whereas the test accuracy was done on a fully untouched set.


### Next steps

- Train for more epochs.
- Test on a different dataset entirely.


## Setup

The notebook installs its own dependencies in the first cell.
Requires an NVIDIA GPU (4-bit quantization needs CUDA); developed and tested on a free
Google Colab T4, connected from VS Code via the official Colab extension.


## Repo structure

```
.
├── finetune_vlm_qlora.ipynb   # main training pipeline (GPU, QLoRA)
└── README.md
```

## Acknowledgments

- [Robo2VLM-1](https://huggingface.co/datasets/keplerccc/Robo2VLM-1) dataset
- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) model
- Built with: [transformers](https://github.com/huggingface/transformers),
  [peft](https://github.com/huggingface/peft), [trl](https://github.com/huggingface/trl),
  and [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)