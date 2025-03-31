import torch
from torch import nn
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from datasets import load_dataset

dataset = load_dataset("go_emotions")

dialog_data = load_dataset("daily_dialog", trust_remote_code = True)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def convert_labels(example):
    labels = torch.zeros(27, dtype=torch.float)
    for label in example["labels"]:
        if 0 <= label < 27:
            labels[label] = 1.0
    example["labels"] = labels.tolist()
    return example

dataset = dataset.map(convert_labels)


dataset = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=27, problem_type = "multi_label_classification")


def custom_loss_function(model, inputs, return_outputs=False):
    labels = inputs.pop("labels").float()  # Ensure labels are float
    outputs = model(**inputs)
    logits = outputs.logits

    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, labels)  # No conversion needed

    return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_loss=custom_loss_function  # Use BCEWithLogitsLoss properly
)

trainer.train()
