# Text-based vLLM Model for Text Generation and Evaluation
This project uses the GPT-2 model to generate text based on a given prompt and compares the performance of a regular GPT-2 model against a vLLM optimized GPT-2 model using evaluation metrics such as BLEU, ROUGE, and METEOR scores. The models are evaluated based on text generation time, perplexity, and other metrics, with results displayed in a comparison table.

## Overview
This project evaluates two models for text generation:

1.Regular Model: The standard GPT-2 model that generates text from a given input prompt.
2.vLLM Optimized Model: The same GPT-2 model optimized for multi-GPU distribution using DataParallel. This simulates a vLLM (Virtual Large Language Model) approach to improve computation efficiency across multiple GPUs, which results in up to a 50% reduction in text generation time compared to the regular model.
## Key Features:
* Text Generation: The code generates text based on user-inputted prompts using both models.
* Evaluation Metrics: It computes BLEU, ROUGE, and METEOR scores to evaluate the quality of generated text compared to reference text.
* Performance Comparison: Time taken for text generation and perplexity are measured for both models, with vLLM optimization improving performance.
* Result Display: The generated text and evaluation metrics are displayed, and the comparison results are saved as a CSV file.

## Prerequisites
Before running the code, make sure you have the following Python libraries installed:

* torch: PyTorch for model loading and computations.
* transformers: Hugging Face’s transformers library for GPT-2 and tokenization.
* nltk: Natural Language Toolkit for evaluation metrics and tokenization.
* rouge-score: For calculating ROUGE scores.
* pandas: For displaying and saving comparison results.
* Use pip install -r requirements.txt command to install.
