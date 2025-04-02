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
from sklearn.metrics import accuracy_score, f1_score
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
        default="distilroberta-base",  # Smaller model for faster training
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
        default=32,  # Increased batch size for faster training
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,  # Reduced to just 1 epoch
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,  # Reduced sequence length
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--dataset_fraction",
        type=float,
        default=0.1,  # Only use 10% of the dataset
        help="Fraction of the dataset to use (0.0-1.0)"
    )
    return parser.parse_args()

def load_goemotions_dataset(dataset_fraction=0.1):
    """Load and prepare the GoEmotions dataset."""
    logger.info(f"Loading {dataset_fraction*100}% of GoEmotions dataset...")

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

    # Reduce dataset size
    train_size = int(len(dataset["train"]) * dataset_fraction)
    validation_size = int(len(dataset["validation"]) * dataset_fraction)
    test_size = int(len(dataset["test"]) * dataset_fraction)

    logger.info(f"Reduced train size: {train_size}, validation size: {validation_size}, test size: {test_size}")

    # Sample the datasets
    reduced_dataset = {
        "train": dataset["train"].select(range(train_size)),
        "validation": dataset["validation"].select(range(validation_size)),
        "test": dataset["test"].select(range(test_size))
    }

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
    processed_dataset = {}
    for split in reduced_dataset:
        processed_dataset[split] = reduced_dataset[split].map(
            preprocess_function,
            batched=True,
            remove_columns=dataset[split].column_names,
        )

    # Map from multi-label to single-label for simplicity
    def convert_to_single_label(example):
        if sum(example["labels"]) == 0:
            # If no emotion, default to neutral (last index)
            example["label"] = len(emotions) - 1
        else:
            # Take the emotion with highest confidence
            example["label"] = np.argmax(example["labels"])
        return example

    single_label_dataset = {}
    for split in processed_dataset:
        single_label_dataset[split] = processed_dataset[split].map(convert_to_single_label)

    return single_label_dataset, emotions

def load_dailydialog_dataset(dataset_fraction=0.1):
    """Load and prepare the DailyDialog dataset."""
    logger.info(f"Loading {dataset_fraction*100}% of DailyDialog dataset...")

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
    processed_dataset = {}
    for split in dataset:
        processed_split = dataset[split].map(
            extract_utterances,
            batched=True,
            remove_columns=dataset[split].column_names
        )

        # Reduce dataset size
        dataset_size = int(len(processed_split) * dataset_fraction)
        processed_dataset[split] = processed_split.select(range(dataset_size))
        logger.info(f"Reduced {split} size: {dataset_size}")

    return processed_dataset, emotions

def compute_metrics(eval_pred):
    """Compute metrics for evaluation using scikit-learn."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)

    # Calculate F1 score (macro average for multi-class)
    f1 = f1_score(labels, predictions, average="macro")

    return {
        "accuracy": accuracy,
        "f1": f1,
    }

def fine_tune_model(args):
    """Fine-tune a pre-trained model on the selected dataset."""
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    if args.dataset == "goemotions":
        dataset, label_names = load_goemotions_dataset(args.dataset_fraction)
    else:  # dailydialog
        dataset, label_names = load_dailydialog_dataset(args.dataset_fraction)

    num_labels = len(label_names)
    logger.info(f"Number of emotion labels: {num_labels}")

    # Load tokenizer and model
    logger.info(f"Loading model: {args.model_name}")
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

    tokenized_dataset = {}
    for split in dataset:
        tokenized_dataset[split] = dataset[split].map(tokenize_function, batched=True)

    # Prepare data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define training arguments with more logging and savings
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=100,  # Evaluate every 100 steps
        save_strategy="steps",
        save_steps=100,   # Save every 100 steps
        save_total_limit=2,  # Only keep the 2 most recent checkpoints
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="none",  # Disable wandb, tensorboard etc.
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,  # Log every 10 steps for more visibility
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
    logger.info("Starting model fine-tuning with reduced dataset...")
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
    logger.info("Since we used a reduced dataset and fewer epochs, the model may not be optimal.")
    logger.info("However, it should still provide reasonable results for your Chrome extension.")

if __name__ == "__main__":
    main()