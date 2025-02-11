import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import load_dataset

MODEL_NAME = "bert-base-multilingual-cased"
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 2e-5
MAX_LENGTH = 512

dataset = load_dataset("csv", data_files={"train": "tr.csv", "test": "test.csv"})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])

train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=BATCH_SIZE)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

for epoch in range(EPOCHS):
    train_loss = train(model, train_dataloader, optimizer)
    test_loss, test_accuracy = evaluate(model, test_dataloader)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

model.save_pretrained("text_classification_model")
tokenizer.save_pretrained("text_classification_model")