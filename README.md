# Creating a Custom Dataset and Fine-tuning SegFormer for Semantic Segmentation

This repository contains notebooks designed to help you upload a custom dataset to Hugging Face Hub, fine-tune a SegFormer model for semantic segmentation, and then push your fine-tuned model to Hugging Face. The example provided focuses on segmentation of plant roots, but the methodology can be applied to any segmentation task.

## Prerequisites

Before you begin, ensure you have:
- A Hugging Face account. Create one at [Hugging Face](https://huggingface.co/join).
- Google Drive space to store your dataset.

## Creating Your Dataset

1. **Prepare Your Images**: The example dataset consisted of `.jpg` images and their segmentation masks, `.png` binary images with pixels in the labeled regions set to '1'. Images and corresponding masks were saved under the same name in separate folders.

3. **Helpful Notebooks for Mask Creation**:
   - K-Means Segmentation for Creating Binary Masks: [Notebook](https://github.com/jacquelinegrimm/kmeans-segmentation/blob/main/arabidopsis_root_segmentation_kmeans.ipynb)
   - Convert to Binary Mask: [Notebook](https://github.com/jacquelinegrimm/useful-scripts/blob/main/convert_to_binary.ipynb)
   These notebooks can assist in generating the masks required for your dataset.

4. **Upload to Google Drive**: Once your dataset is prepared, upload the images and masks to Google Drive.

## Uploading Dataset to Hugging Face Hub

1. **hf_custom_dataset.ipynb**: Use the provided Colab notebook `hf_custom_dataset.ipynb` to create a custom dataset and push it to Hugging Face Hub.

2. **Create an `id2label.json` File**: After pushing your data to Hugging Face, you'll need to create an `id2label.json` file that maps your dataset's labels to their respective IDs. Upload this file to your dataset's repository on Hugging Face. An example `id2label.json` is available in this repo for reference.

## Fine-tuning the SegFormer Model

1. **finetune_segformer_model.ipynb**: This notebook guides you through fine-tuning a SegFormer model and pushing it to Hugging Face Hub. This specific example is for second-stage fine-tuning of a SegFormer model for segmentation of plant roots. To start instead from the base model, replace the model name in the code with `"nvidia/mit-b0"`.

2. **Model and Fine-tuning Resources**:
   - Base Model: [nvidia/mit-b0](https://huggingface.co/nvidia/mit-b0)
   - Example Fine-tuned Model: [segformer-b0-finetuned-arabidopsis-roots-v03](https://huggingface.co/jacquelinegrimm/segformer-b0-finetuned-arabidopsis-roots-v03)
