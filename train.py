import os
import torch
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
CSV_FILE_PATH = "abc.csv"  # Ensure this file is in your running directory

# 2. Load and Prepare CSV Data
if not os.path.exists(CSV_FILE_PATH):
    raise FileNotFoundError(f"Could not find {CSV_FILE_PATH}. Please check the path.")

# Load CSV via Hugging Face Datasets
raw_dataset = load_dataset("csv", data_files=CSV_FILE_PATH, split="train")

# Filter out any accidental empty/NaN rows in your CSV
raw_dataset = raw_dataset.filter(lambda x: x["English"] is not None and x["Urdu"] is not None)

# 3. Load Tokenizer and Base Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

# 4. Define Data Preprocessing Function
def preprocess_function(examples):
    # Match the exact column names from your abc.csv file
    inputs = examples["English"]
    targets = examples["Urdu"]
    
    # Tokenize English inputs
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    
    # Tokenize Urdu targets
    labels = tokenizer(text_target=targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Process the dataset and remove the original raw text columns
tokenized_dataset = raw_dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=["English", "Urdu"]
)

# 5. Apply LoRA Configuration for Seq2Seq Models
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,  
    inference_mode=False,
    r=16,                             
    lora_alpha=32,                    
    lora_dropout=0.1,                 
    target_modules=["q_proj", "v_proj"] 
)

# Wrap the base model with LoRA layers
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  

# 6. Configure Training Parameters
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,    
    gradient_accumulation_steps=2,    
    learning_rate=5e-4,               
    num_train_epochs=5,               
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no",         
    fp16=torch.cuda.is_available(),   
    predict_with_generate=True
)

# 7. Initialize Trainer
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 8. Run Training and Save Output
print(f"Loaded {len(raw_dataset)} sentences from {CSV_FILE_PATH}. Starting LoRA fine-tuning...")
trainer.train()

# Save the small LoRA adapter weights (approx 5-10 MB)
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Training complete! Adapter weights saved to {OUTPUT_DIR}")
