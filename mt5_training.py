import time
import pandas as pd
import threading
import itertools
import sys
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate
import numpy as np
import nltk
from tqdm.auto import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "bilingual_books_filtered_titles.parquet"
MODEL_CHECKPOINT = "google/mt5-small"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 30
BATCH_SIZE = 4
EPOCHS = 2
OUTPUT_DIR = "./mt5_summarization_results"

# ============================================================
# VERSION CHECK (useful debug info)
# ============================================================

import transformers, datasets, tokenizers
print(f"✅ Transformers: {transformers.__version__}")
print(f"✅ Datasets: {datasets.__version__}")
print(f"✅ Tokenizers: {tokenizers.__version__}")
print(f"✅ Evaluate: {evaluate.__version__}")

# ============================================================
# UTILITY TIMER
# ============================================================

def timed_stage(label, func, *args, **kwargs):
    """Helper to measure runtime of each stage."""
    print(f"\n⏳ {label} ...")
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    print(f"✅ {label} completed in {elapsed:.2f} seconds.\n")
    return result

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

def load_data():
    print(f"📦 Loading data from {DATA_FILE} ...")
    df = pd.read_parquet(DATA_FILE)

    df = df.dropna(subset=["review_body", "review_title"]).reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_end = int(0.8 * len(df))
    val_end = int(0.9 * len(df))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"✅ Train: {len(train_df):,}, Validation: {len(val_df):,}, Test: {len(test_df):,}")

    return (
        Dataset.from_pandas(train_df),
        Dataset.from_pandas(val_df),
        Dataset.from_pandas(test_df),
        test_df,
    )

train_dataset, val_dataset, test_dataset, test_df = timed_stage("Loading and preparing data", load_data)

# ============================================================
# LOAD TOKENIZER AND MODEL (with spinner)
# ============================================================

def load_model():
    print(f"🔤 Loading tokenizer and model: {MODEL_CHECKPOINT}")

    done = False

    def animate():
        for c in itertools.cycle(['|', '/', '-', '\\']):
            if done:
                break
            sys.stdout.write(f'\r⌛ Loading tokenizer and model... {c}')
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\r✅ Tokenizer and model loaded successfully!     \n')

    t = threading.Thread(target=animate)
    t.start()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    done = True
    t.join()
    return tokenizer, model

tokenizer, model = timed_stage("Loading tokenizer and model", load_model)

# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["review_body"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )
    labels = tokenizer(
        examples["review_title"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# ============================================================
# TOKENIZATION (with progress bars + multiprocessing)
# ============================================================

def tokenize_dataset():
    from datasets.utils.logging import set_verbosity_warning
    set_verbosity_warning()

    print("🧮 Tokenizing datasets (with progress bars)...")

    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=64,
        num_proc=2,
        desc="Tokenizing training set",
    )

    tokenized_val = val_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=64,
        num_proc=2,
        desc="Tokenizing validation set",
    )

    return tokenized_train, tokenized_val

tokenized_train, tokenized_val = timed_stage("Tokenization", tokenize_dataset)

# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ============================================================
# LOAD ROUGE METRIC
# ============================================================

rouge = evaluate.load("rouge")
nltk.download("punkt", quiet=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: v.mid.fmeasure * 100 for k, v in result.items()}
    return {k: round(v, 4) for k, v in result.items()}

# ============================================================
# TRAINING ARGUMENTS (Seq2Seq)
# ============================================================

args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    predict_with_generate=True,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=100,
    save_total_limit=3,
    disable_tqdm=False,
    report_to="none",
)

# ============================================================
# SEQ2SEQ TRAINER
# ============================================================

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ============================================================
# TRAIN
# ============================================================

timed_stage("Fine-tuning mT5 model", trainer.train)

# ============================================================
# EVALUATE ON TEST SET
# ============================================================

def evaluate_model():
    print("\n📊 Evaluating on test set ...")
    tokenized_test = test_dataset.map(preprocess_function, batched=True, desc="Tokenizing test set")
    metrics = trainer.evaluate(tokenized_test)
    print("ROUGE scores on test data:", metrics)

timed_stage("Evaluation", evaluate_model)

# ============================================================
# GENERATE SAMPLE SUMMARIES
# ============================================================

def generate_samples():
    print("\n🧾 Generating sample summaries ...")
    sample_texts = test_df["review_body"].sample(3, random_state=42).tolist()

    for i, text in enumerate(sample_texts, 1):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH)
        outputs = model.generate(**inputs, max_length=MAX_TARGET_LENGTH)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n🔹 Sample {i}")
        print(f"Review: {text[:250]}...")
        print(f"Generated title: {summary}")

timed_stage("Generating sample summaries", generate_samples)
