import pandas as pd
import torch
import evaluate
import numpy as np

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

training_dataset="email_dataset_250.csv"

# STEP 1 : Reading the training data
print("Step 1: Load the Dataset")
# Load the dataset
df = pd.read_csv(training_dataset)

# Split into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

print(f"Training examples: {len(train_texts)}")
print(f"Testing examples: {len(test_texts)}")

#STEP 2 : Tokenize the Text using BERT Tokenizer
print("Step 2: Tokenize the Text using BERT Tokenizer")

# Load pre-trained BERT tokenizer
model_name="bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

# Tokenize the dataset
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# STEP 3 : Convert Data to PyTorch Dataset Format
print("Step 3: Convert Data to PyTorch Dataset Format")
class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# Convert to PyTorch Dataset
train_dataset = EmailDataset(train_encodings, train_labels)
test_dataset = EmailDataset(test_encodings, test_labels)


# STEP 4 : Load Pre-Trained BERT Model for Classification
print("Step 4: Load Pre-Trained BERT Model for Classification")

# Load pre-trained BERT model with 6 output labels
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# STEP 5 : Train the Model
training_args = TrainingArguments(
    output_dir="./results",        # Save model checkpoints
    evaluation_strategy="epoch",   # Evaluate after each epoch
    per_device_train_batch_size=8, # Batch size
    per_device_eval_batch_size=8,
    num_train_epochs=3,            # Number of training epochs
    weight_decay=0.01,             # Regularization
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# STEP 6 : Evaluate the Model
print("Step 6: Evaluate the Model")
trainer.evaluate()

# STEP 7 : Save and Load the Fine-Tuned Model
print("step 7: Save and Load the Fine-Tuned Model")
model.save_pretrained("model_fine_tuned_bert_email")
tokenizer.save_pretrained("model_fine_tuned_bert_email")

