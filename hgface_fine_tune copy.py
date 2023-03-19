import os
os.environ['TRANSFORMERS_CACHE'] = '/data2/hanyings/.cache'

import transformers
from datasets import load_dataset, load_metric

import datasets
import random
import pandas as pd
from IPython.display import display, HTML
from transformers import AutoTokenizer, logging
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import wandb
import torch



raw_datasets = load_dataset("wmt17", "zh-en", cache_dir="/data2/hanyings/.cache")
metric = load_metric("sacrebleu")
split = raw_datasets["train"].train_test_split(test_size=10)
raw_datasets["train"] = split["test"]

def show_random_elements(dataset, num_examples=5):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))
show_random_elements(raw_datasets["train"])

model_checkpoint = "/data2/hanyings/opus-mt-en-zh-finetuned-en-to-zh/checkpoint-2063500"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")

# preprocess 
prefix = ""
max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "zh"
def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
print("tokenize datasets")
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
print(tokenized_datasets)

#WandB
wandb.init(name=f'opus-mt-en-zh-finetuned-{source_lang}-to-{target_lang}-1109', 
           project='huggingface',
           notes='Fine tune the transformer model, Helsinki-NLP/opus-mt-en-zh', 
           tags=['huggingface', 'transformer'])

print("tokenization done")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)
model.gradient_checkpointing_enable()
print(next(model.parameters()).is_cuda)
wandb.watch(model)


batch_size = 32
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"opus-mt-en-zh-finetuned-{source_lang}-to-{target_lang}-1109",
    evaluation_strategy = "epoch", # evaluate on valid dataset at emd pf each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=20,
    predict_with_generate=True,
    metric_for_best_model = "bleu",  #Must be the name of a metric returned by the evaluation with or without the prefix "eval_".
    # bleu in the compute metric ?   
    gradient_checkpointing =True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

import numpy as np
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, tokenize="zh",smooth_method="add-k")
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()