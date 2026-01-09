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

## What the Code Does
* Text Generation: The code uses the GPT-2 model from the transformers library to generate text based on an input prompt.
A regular GPT-2 model generates text, and a simulated vLLM optimized model generates text using DataParallel for multiple GPU usage.

* Evaluation: The generated text is compared to a reference text using BLEU, ROUGE, and METEOR scores.
The time taken for text generation and perplexity (a measure of how well the model predicts the text) are also computed.

* Comparison: A comparison table is created that includes the BLEU, ROUGE, and METEOR scores for both models. The results are printed in the console and saved to a CSV file.

## Output:
<img width="1470" alt="Screenshot 2025-02-17 at 5 56 51 PM" src="https://github.com/user-attachments/assets/62988db9-f7fd-4292-8bfd-ed92a722612c" />

![image](https://github.com/user-attachments/assets/6f29890e-83db-4e39-a163-f31a21585fe9)



## Performance Impact and Optimization
### vLLM Optimization: By using DataParallel for multi-GPU processing, the vLLM optimized model provides a significant performance improvement, including:

Up to 50% faster text generation time: The vLLM model processes requests much faster due to distributed computation across multiple GPUs, compared to the regular model that runs on a single GPU (or CPU).
This speed boost makes the model suitable for real-time applications or large-scale text generation tasks.
## Future Enhancements
* This project can be extended or improved in several ways:

* Model Fine-Tuning: Fine-tune the GPT-2 model on specific datasets to improve the quality of generated text for particular domains (e.g., healthcare, finance, etc.).
  
* Performance Optimization: Experiment with different model architectures (like GPT-3 or T5) to improve text generation performance. Further optimize the vLLM model for multi-GPU environments to handle even larger models or datasets.

* Evaluation with More Metrics: Include additional metrics like ROUGE-L or F1-score for a more comprehensive comparison. Implement a human evaluation module for better understanding of model outputs.

* Interactive Interface: Build a web interface or GUI that allows users to input text and reference text interactively, with visual feedback on the evaluation metrics. Implement an API that can be accessed for real-time text generation and evaluation.

* Handling Long Texts: Modify the models and tokenization process to handle longer texts more efficiently by breaking them into manageable chunks. Implement document summarization for large PDFs or articles to generate summaries using the GPT-2 model.

#Ajith
#Ajith
# Ajith