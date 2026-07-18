import os
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# 1. Configuration Constants
BASE_MODEL_ID = "Helsinki-NLP/opus-mt-en-ur"
LORA_ADAPTER_DIR = "./opus-mt-en-ur-lora"
CSV_FILE_PATH = "test.csv"

# 2. Check for Trained Adapter Weights
if not os.path.exists(LORA_ADAPTER_DIR):
    raise FileNotFoundError(
        f"Could not find the adapter folder at '{LORA_ADAPTER_DIR}'. "
        "Please run your training script first."
    )

# 3. Load 10 Random Sentences from your CSV
print(f"Reading sentences from {CSV_FILE_PATH}...")
raw_dataset = load_dataset("csv", data_files=CSV_FILE_PATH, split="train")

# Filter out empty entries and pull the 'English' column texts
english_sentences = [
    row["English"] for row in raw_dataset 
    if row["English"] is not None and str(row["English"]).strip() != ""
]

# Randomly select 10 unique sentences
sample_size = min(10, len(english_sentences))
random_samples = random.sample(english_sentences, sample_size)

# 4. Load Base Model, Tokenizer, and Apply LoRA Weights
print("Loading model and applying fine-tuned LoRA adapters...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID)

# Wrap base model with your LoRA configurations
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)

# Move model to GPU if available for faster speed
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()  # Put model into evaluation mode

print(f"\n--- Translating {sample_size} Random Sentences ---")

# 5. Run Inference Loop
with torch.no_grad():  # Disable gradient tracking to save RAM
    for idx, sentence in enumerate(random_samples, 1):
        # Prepare input tokens
        inputs = tokenizer(sentence, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translated tokens
        generated_tokens = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,       # Use beam search for higher translation quality
            early_stopping=True
        )
        
        # Decode tokens back into readable Urdu text
        translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
        # Print results cleanly
        print(f"\n[{idx}] English: {sentence}")
        print(f"    Urdu Fine-Tuned Translation: {translation}")

print("\nInference complete!")
