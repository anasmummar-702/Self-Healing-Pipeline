"""
This script demonstrates the fine-tuning process for DistilBERT on the SST-2 dataset.
(Note: The final trained model weights are saved in the local_model/ directory).
"""
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def fine_tune_model():
    print("Loading dataset...")
    dataset = load_dataset("glue", "sst2")
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir="../local_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )
    
    print("Starting Fine-Tuning...")
    trainer.train()
    trainer.save_model("../local_model")
    print("Model saved to ../local_model")

if __name__ == "__main__":
    print("Fine-tuning script created for demonstration purposes.")
    # fine_tune_model() # Uncomment to actually run the training