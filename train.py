from transformers import LlamaForCausalLM, PreTrainedTokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset

# Load your data
data_files = {"train": "data/train_modified.json", "test": "data/test_modified.json"}
dataset = load_dataset("json", data_files=data_files)

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize data
def tokenize_function(examples):
    return tokenizer(examples["dialogue"], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Fine-tuning setup
training_args = TrainingArguments(
    output_dir="./output",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    fp16=True,  # Use FP16 for larger models if GPU allows
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

# Train the model
trainer.train()
