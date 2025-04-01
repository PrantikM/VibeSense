import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import torch
from torch.utils.data import DataLoader
import evaluate
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a transformer model on sentiment analysis datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["goemotions", "dailydialog"],
        help="Dataset to fine-tune on"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-base",
        help="Model identifier from Hugging Face model hub"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./fine-tuned-model",
        help="Directory to save the fine-tuned model"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    return parser.parse_args()

def load_goemotions_dataset():
    """Load and prepare the GoEmotions dataset."""
    logger.info("Loading GoEmotions dataset...")

    # Load the dataset from Hugging Face datasets
    dataset = load_dataset("go_emotions")

    # Get the emotion labels
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]

    # Convert multi-label format to the format needed for training
    def preprocess_function(examples):
        # Create one-hot encoding for the emotion labels
        labels = np.zeros((len(examples["text"]), len(emotions)), dtype=np.float32)

        for i, example_labels in enumerate(examples["labels"]):
            for label in example_labels:
                labels[i][label] = 1.0

        return {
            "text": examples["text"],
            "labels": labels.tolist(),
        }

    # Apply preprocessing
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Map from multi-label to single-label for simplicity in this example
    # (in practice, you might want to keep it multi-label)
    def convert_to_single_label(example):
        if sum(example["labels"]) == 0:
            # If no emotion, default to neutral (last index)
            example["label"] = len(emotions) - 1
        else:
            # Take the emotion with highest confidence
            example["label"] = np.argmax(example["labels"])
        return example

    single_label_dataset = processed_dataset.map(convert_to_single_label)

    return single_label_dataset, emotions

def load_dailydialog_dataset():
    """Load and prepare the DailyDialog dataset."""
    logger.info("Loading DailyDialog dataset...")

    # Load the dataset from Hugging Face datasets
    dataset = load_dataset("daily_dialog")

    # The emotions in DailyDialog are coded as integers:
    # 0: neutral, 1: happiness, 2: surprise, 3: sadness,
    # 4: anger, 5: disgust, 6: fear
    emotions = [
        'neutral', 'happiness', 'surprise', 'sadness',
        'anger', 'disgust', 'fear'
    ]

    # Extract utterances and their emotion labels
    def extract_utterances(examples):
        texts = []
        labels = []

        # Each dialog has a list of utterances and a list of emotions
        for dialog, emotions_list in zip(examples["dialog"], examples["emotion"]):
            for utterance, emotion in zip(dialog, emotions_list):
                texts.append(utterance)
                labels.append(emotion)

        return {"text": texts, "label": labels}

    # Apply preprocessing
    processed_dataset = dataset.map(
        extract_utterances,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    return processed_dataset, emotions

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy_metric = evaluate.load("accuracy")
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)

    f1_metric = evaluate.load("f1")
    # For multi-class, use macro average
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
    }

def fine_tune_model(args):
    """Fine-tune a pre-trained model on the selected dataset."""
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    if args.dataset == "goemotions":
        dataset, label_names = load_goemotions_dataset()
    else:  # dailydialog
        dataset, label_names = load_dailydialog_dataset()

    num_labels = len(label_names)
    logger.info(f"Number of emotion labels: {num_labels}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        problem_type="single_label_classification"
    )

    # Tokenize the datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Prepare data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="none",  # Disable wandb, tensorboard etc.
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    logger.info("Starting model fine-tuning...")
    trainer.train()

    # Evaluate the model
    logger.info("Evaluating the model...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")

    # Save the model, tokenizer, and configuration
    logger.info(f"Saving the model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save label names for inference
    with open(os.path.join(args.output_dir, "label_names.txt"), "w") as f:
        for label in label_names:
            f.write(f"{label}\n")

    return args.output_dir

def main():
    args = parse_args()
    fine_tuned_model_path = fine_tune_model(args)
    logger.info(f"Model fine-tuning complete. Model saved to: {fine_tuned_model_path}")

if __name__ == "__main__":
    main()