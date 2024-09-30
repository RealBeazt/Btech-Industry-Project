import pandas as pd
import datasets
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder

train_single_df = pd.read_csv('Dataset/df_single_train.csv')
train_multi_df = pd.read_csv('Dataset/df_multi_train.csv')
dev_single_df = pd.read_csv('Dataset/df_single_dev.csv')
dev_multi_df = pd.read_csv('Dataset/df_multi_dev.csv')
test_df = pd.read_csv('Dataset/df_test.csv')

train_df = pd.concat([train_single_df, train_multi_df], axis=0)
dev_df = pd.concat([dev_single_df, dev_multi_df], axis=0)

le = LabelEncoder()
train_df['label'] = le.fit_transform(train_df['label'])
dev_df['label'] = le.transform(dev_df['label'])
test_df['label'] = le.transform(test_df['label'])

train_data = datasets.Dataset.from_pandas(train_df)
dev_data = datasets.Dataset.from_pandas(dev_df)
test_data = datasets.Dataset.from_pandas(test_df)

# Initialize tokenizer
model_checkpoint = "bert-base-uncased"  # or any transformer model suitable for your use case
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Tokenize datasets
train_dataset = train_data.map(tokenize_function, batched=True)
dev_dataset = dev_data.map(tokenize_function, batched=True)
test_dataset = test_data.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
dev_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)  # Adjust batch size as needed
dev_dataloader = DataLoader(dev_dataset, batch_size=8)
test_dataloader = DataLoader(test_dataset, batch_size=8)

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=train_df['label'].nunique())

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
)

# Train the model
trainer.train()

trainer.evaluate()

test_results = trainer.predict(test_dataset)
print(test_results.metrics)