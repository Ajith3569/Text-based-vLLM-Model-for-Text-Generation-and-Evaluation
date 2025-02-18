import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import pandas as pd
import pandas as pd

nltk.download('wordnet')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "gpt2-large"  
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  
model_regular = GPT2LMHeadModel.from_pretrained(model_name)
model_regular.to(device)  

model_vllm = GPT2LMHeadModel.from_pretrained(model_name)
model_vllm.to(device)

if torch.cuda.device_count() > 1:
    model_vllm = torch.nn.DataParallel(model_vllm)

# Function to generate text based on a prompt for Regular Model
def generate_text_regular(prompt, max_length=1000):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)  
    attention_mask = inputs["attention_mask"].to(device)  
    
    outputs = model_regular.generate(input_ids, 
                                     attention_mask=attention_mask,  
                                     max_length=max_length, 
                                     num_return_sequences=1)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Function to generate text based on a prompt for vLLM Optimized Model
def generate_text_vllm(prompt, max_length=1000):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)  
    attention_mask = inputs["attention_mask"].to(device) 
    
    model = model_vllm.module if isinstance(model_vllm, torch.nn.DataParallel) else model_vllm
    
    outputs = model.generate(input_ids, 
                             attention_mask=attention_mask,  
                             max_length=max_length, 
                             num_return_sequences=1)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Function to calculate Perplexity
def calculate_perplexity(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()  # The model uses the same sequence as input and target

    # Move both input and labels to the same device as the model
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)  # Ensure both input and labels are on the same device
        loss = outputs.loss
    
    perplexity = torch.exp(loss)
    return perplexity.item()

# user input and reference text at runtime
user_input = input("Please enter a prompt for text generation: ")
reference_text = input("Please enter the reference text for evaluation: ")

# Measure time for Regular Model
start = time.time()
generated_text_regular = generate_text_regular(user_input)
regular_time = time.time() - start

# Measure time for vLLM Optimized Model
start_vllm = time.time()
generated_text_vllm = generate_text_vllm(user_input)
vllm_time = time.time() - start_vllm

# Evaluate Regular Model
perplexity_regular = calculate_perplexity(user_input, model_regular, tokenizer)

# Evaluate vLLM Optimized Model
perplexity_vllm = calculate_perplexity(user_input, model_vllm, tokenizer)

# Compute BLEU, ROUGE, and METEOR scores for both models
def calculate_bleu(reference, candidate):
    return sentence_bleu([reference], candidate.split())

def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

def calculate_meteor(reference, candidate):
    reference_tokens = word_tokenize(reference)
    candidate_tokens = word_tokenize(candidate)
    return meteor_score([reference_tokens], candidate_tokens)

# Compute evaluation metrics
bleu_score_regular = calculate_bleu([reference_text], generated_text_regular)
rouge_score_regular = calculate_rouge(reference_text, generated_text_regular)
meteor_score_regular = calculate_meteor(reference_text, generated_text_regular)

bleu_score_vllm = calculate_bleu([reference_text], generated_text_vllm)
rouge_score_vllm = calculate_rouge(reference_text, generated_text_vllm)
meteor_score_vllm = calculate_meteor(reference_text, generated_text_vllm)

# Function to create the comparison table
def create_comparison_table(bleu_score_regular, rouge_score_regular, meteor_score_regular, 
                            bleu_score_vllm, rouge_score_vllm, meteor_score_vllm):
    data = {
        "Metric": ["BLEU Score", "ROUGE Score", "METEOR Score"],
        "Regular Model": [bleu_score_regular, rouge_score_regular, meteor_score_regular],
        "vLLM Optimized Model": [bleu_score_vllm, rouge_score_vllm, meteor_score_vllm],
    }
    
    df = pd.DataFrame(data)
    return df

# Create the comparison table
comparison_table = create_comparison_table(bleu_score_regular, rouge_score_regular, meteor_score_regular, 
                                           bleu_score_vllm, rouge_score_vllm, meteor_score_vllm)

# Display the generated outputs and comparison table
print(f"\nGenerated Text (Regular): {generated_text_regular}")
print(f"Time taken by Regular Model: {regular_time} seconds")
print(f"Perplexity (Regular): {perplexity_regular}")

print(f"\nGenerated Text (vLLM Optimized): {generated_text_vllm}")
print(f"Time taken by vLLM Optimized Model: {vllm_time} seconds")
print(f"Perplexity (vLLM Optimized): {perplexity_vllm}")


# Show the comparison table in the console
print("\nComparison of Evaluation Metrics:")
print(comparison_table)

# Optionally, save it to a CSV file if you need
comparison_table.to_csv("comparison_results.csv", index=False)
print("\nComparison results saved to comparison_results.csv")

# Show the comparison table in the console as well
print("\nComparison of Evaluation Metrics:")
print(comparison_table)
