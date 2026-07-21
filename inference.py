import os
import random
import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# 1. Configuration Constants
BASE_MODEL_ID = "Helsinki-NLP/opus-mt-en-ur"
LORA_ADAPTER_DIR = "./opus-mt-en-ur-lora"
CSV_FILE_PATH = "test.csv"
SEED = 42  # NEW: reproducible sampling

# 2. Check for Trained Adapter Weights
if not os.path.exists(LORA_ADAPTER_DIR):
    raise FileNotFoundError(
        f"Could not find the adapter folder at '{LORA_ADAPTER_DIR}'. "
        "Please run your training script first."
    )

# 3. Load sentence PAIRS from your CSV (English + Urdu reference)
print(f"Reading sentences from {CSV_FILE_PATH}...")
raw_dataset = load_dataset("csv", data_files=CSV_FILE_PATH, split="train")

# --- CHANGED: keep English/Urdu paired instead of only grabbing English ---
pairs = [
    (row["English"], row["Urdu"])
    for row in raw_dataset
    if row["English"] is not None and str(row["English"]).strip() != ""
    and row["Urdu"] is not None and str(row["Urdu"]).strip() != ""
]

random.seed(SEED)
sample_size = min(10, len(pairs))
random_samples = random.sample(pairs, sample_size)

# 4. Load Base Model, Tokenizer, and Apply LoRA Weights
print("Loading model and applying fine-tuned LoRA adapters...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID)

model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# --- NEW: BLEU metric setup ---
bleu_metric = evaluate.load("sacrebleu")
predictions = []
references = []

print(f"\n--- Translating {sample_size} Random Sentences ---")

# 5. Run Inference Loop
with torch.no_grad():
    for idx, (sentence, reference) in enumerate(random_samples, 1):
        inputs = tokenizer(sentence, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generated_tokens = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

        translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        # --- NEW: per-sentence BLEU ---
        sentence_bleu = bleu_metric.compute(
            predictions=[translation],
            references=[[reference]]
        )["score"]

        predictions.append(translation)
        references.append([reference])

        print(f"\n[{idx}] English: {sentence}")
        print(f"    Reference Urdu:      {reference}")
        print(f"    Fine-Tuned Translation: {translation}")
        print(f"    Sentence BLEU: {sentence_bleu:.2f}")

# --- NEW: overall corpus BLEU across all sampled sentences ---
corpus_bleu = bleu_metric.compute(predictions=predictions, references=references)["score"]

print("\n" + "=" * 50)
print(f"Corpus BLEU score over {sample_size} sentences: {corpus_bleu:.2f}")
print("=" * 50)

print("\nInference complete!")
