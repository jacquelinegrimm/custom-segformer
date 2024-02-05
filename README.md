# Creating a Custom Dataset and Fine-tuning SegFormer for Semantic Segmentation

This repository contains notebooks designed to help you upload a custom dataset to Hugging Face Hub, fine-tune a SegFormer model, and push your fine-tuned model to Hugging Face. The example provided focuses on segmentation of plant roots, but the methodology can be applied to any semantic segmentation task.

## Prerequisites

Before you begin, ensure you have:
- A Hugging Face account. Create one at [Hugging Face](https://huggingface.co/join).

## Creating Your Dataset

1. **Prepare Your Images**: The example dataset consisted of `.jpg` images and their segmentation masks, `.png` binary images with pixels in the labeled regions set to '1'. Images and corresponding masks were saved under the same name in separate folders.

3. **Helpful Notebooks for Mask Creation**: These notebooks may assist in generating segmentation masks:
   - K-Means Segmentation for Creating Binary Masks: [Notebook](https://github.com/jacquelinegrimm/kmeans-segmentation/blob/main/arabidopsis_root_segmentation_kmeans.ipynb)
   - Convert to Binary Mask: [Notebook](https://github.com/jacquelinegrimm/useful-scripts/blob/main/convert_to_binary.ipynb)

4. **Upload to Google Drive**: Once your dataset is prepared, upload the images and masks to Google Drive.

## Uploading Dataset to Hugging Face Hub

1. **hf_custom_dataset.ipynb**: Use the provided Colab notebook `hf_custom_dataset.ipynb` to create a custom dataset and push it to Hugging Face Hub.

2. **Create an `id2label.json` File**: After pushing your dataset to Hugging Face, you'll need to create an `id2label.json` file that maps your dataset's labels to their respective IDs. Upload this file to your dataset's repository on Hugging Face. An example `id2label.json` is available in this repo for reference.

## Fine-tuning the SegFormer Model

1. **finetune_segformer_model.ipynb**: This notebook guides you through the process of fine-tuning a SegFormer model and pushing it to Hugging Face Hub. This example shows second-stage fine-tuning of a model for segmentation of plant root images. To start instead from the original pre-trained model, replace the model name in the code with `"nvidia/mit-b0"`.

2. **Model and Fine-tuning Resources**:
   - SegFormer Model: [nvidia/mit-b0](https://huggingface.co/nvidia/mit-b0)
   - Example Fine-tuned Model: [segformer-b0-finetuned-arabidopsis-roots-v03](https://huggingface.co/jacquelinegrimm/segformer-b0-finetuned-arabidopsis-roots-v03)
