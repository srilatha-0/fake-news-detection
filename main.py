# distilbert_fake_news_subset.py

import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import streamlit as st
import os
from sklearn.utils import shuffle

# -------------------------------
# 1️⃣ Load and Sample Dataset
# -------------------------------
def load_sampled_dataset(sample_size=5000):
    # Load CSVs
    true_csv = pd.read_csv("datasets/True.csv")
    fake_csv = pd.read_csv("datasets/Fake.csv")
    
    true_csv['label'] = 1
    fake_csv['label'] = 0
    
    # Shuffle and sample
    true_sample = shuffle(true_csv).head(sample_size)
    fake_sample = shuffle(fake_csv).head(sample_size)
    
    df = pd.concat([true_sample, fake_sample]).reset_index(drop=True)
    df = shuffle(df).reset_index(drop=True)
    return df

# -------------------------------
# 2️⃣ Prepare Dataset for DistilBERT
# -------------------------------
def prepare_dataset(df):
    dataset = Dataset.from_pandas(df[['title', 'label']])
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Tokenize
    def tokenize(batch):
        return tokenizer(batch['title'], padding=True, truncation=True, max_length=128)
    
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Split train/test
    train_test = dataset.train_test_split(test_size=0.2)
    return train_test['train'], train_test['test'], tokenizer

# -------------------------------
# 3️⃣ Train DistilBERT Model
# -------------------------------
def train_model(train_dataset, eval_dataset):
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=2
    )

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="steps",
        save_steps=50,
        logging_steps=10,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()
    trainer.save_model('./models/distilbert_fake_news')
    return model

# -------------------------------
# 4️⃣ Prediction Function
# -------------------------------
def predict_news(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits).item()
    return "🟢 Real News" if pred == 1 else "🔴 Fake News"

# -------------------------------
# 5️⃣ Streamlit UI
# -------------------------------
def run_streamlit():
    st.title("📰 Fake News Detector (DistilBERT - Subset Training)")
    st.write("Enter a news headline to check if it's Real or Fake.")

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    # Load trained model or train on subset
    if os.path.exists('./models/distilbert_fake_news'):
        model = DistilBertForSequenceClassification.from_pretrained('./models/distilbert_fake_news')
    else:
        df = load_sampled_dataset(sample_size=5000)  # 5k True + 5k Fake
        train_dataset, eval_dataset, tokenizer = prepare_dataset(df)
        model = train_model(train_dataset, eval_dataset)

    user_input = st.text_area("📝 News Input")
    if st.button("🚀 Predict"):
        if user_input.strip() == "":
            st.warning("Please enter some news text!")
        else:
            result = predict_news(model, tokenizer, user_input)
            st.success(result)

# -------------------------------
# 6️⃣ Main
# -------------------------------
if __name__ == "__main__":
    run_streamlit()