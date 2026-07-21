import os
import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType

# 1. Configuration Constants
MODEL_ID = "Helsinki-NLP/opus-mt-en-ur"
OUTPUT_DIR = "./opus-mt-en-ur-lora"
CSV_FILE_PATH = "abc.csv"

# 2. Load and Prepare CSV Data
if not os.path.exists(CSV_FILE_PATH):
    raise FileNotFoundError(f"Could not find {CSV_FILE_PATH}. Please check the path.")

raw_dataset = load_dataset("csv", data_files=CSV_FILE_PATH, split="train")
raw_dataset = raw_dataset.filter(lambda x: x["English"] is not None and x["Urdu"] is not None)

# --- NEW: split into train/eval since there's no eval set at all ---
# With only 80 sentences, keep the eval split small but non-trivial (~10-15%)
split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
train_raw = split_dataset["train"]
eval_raw = split_dataset["test"]
print(f"Train: {len(train_raw)} sentences | Eval: {len(eval_raw)} sentences")

# 3. Load Tokenizer and Base Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

# 4. Preprocessing
def preprocess_function(examples):
    inputs = examples["English"]
    targets = examples["Urdu"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(text_target=targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_raw.map(preprocess_function, batched=True, remove_columns=["English", "Urdu"])
tokenized_eval = eval_raw.map(preprocess_function, batched=True, remove_columns=["English", "Urdu"])

# 5. LoRA config
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --- NEW: BLEU metric via sacrebleu (the standard for MT) ---
bleu_metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # -100 is the ignore-index used for padding in labels; must restore pad token before decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [[l.strip()] for l in decoded_labels]  # sacrebleu wants list-of-references per prediction

    result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

# 6. Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,          # NEW
    gradient_accumulation_steps=2,
    learning_rate=5e-4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",                 # CHANGED from "no" -> evaluate every epoch
    fp16=torch.cuda.is_available(),
    predict_with_generate=True,
    generation_max_length=128,             # NEW: ensures generated translations aren't cut short
    load_best_model_at_end=True,           # NEW: keep the checkpoint with the best BLEU
    metric_for_best_model="bleu",          # NEW
    greater_is_better=True,                # NEW
)

# 7. Trainer
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,           # NEW
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,       # NEW
)

# 8. Train
print(f"Loaded {len(raw_dataset)} sentences total. Starting LoRA fine-tuning...")
trainer.train()

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Training complete! Adapter weights saved to {OUTPUT_DIR}")
