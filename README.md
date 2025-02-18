# Text-based vLLM Model for Text Generation and Evaluation
This project uses the GPT-2 model to generate text based on a given prompt and compares the performance of a regular GPT-2 model against a vLLM optimized GPT-2 model using evaluation metrics such as BLEU, ROUGE, and METEOR scores. The models are evaluated based on text generation time, perplexity, and other metrics, with results displayed in a comparison table.

## Overview
This project evaluates two models for text generation:

1.Regular Model: The standard GPT-2 model that generates text from a given input prompt.
2.vLLM Optimized Model: The same GPT-2 model optimized for multi-GPU distribution using DataParallel. This simulates a vLLM (Virtual Large Language Model) approach to improve computation efficiency across multiple GPUs, which results in up to a 50% reduction in text generation time compared to the regular model.
