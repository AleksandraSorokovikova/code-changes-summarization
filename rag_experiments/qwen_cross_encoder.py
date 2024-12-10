from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from tqdm.notebook import tqdm
from transformers import AutoModelForSequenceClassification
import torch.nn as nn
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import AutoModelForCausalLM
import torch
import json
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from itertools import combinations
from tqdm import tqdm
import random
import pandas as pd
from datasets import load_dataset


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_embedding(model, tokenizer, code_snippet, device='cpu'):
    with torch.no_grad():
        encoded_input = tokenizer(code_snippet, padding=True, truncation=True, return_tensors='pt').to(device)
        model_output = model(**encoded_input)
        embedding = mean_pooling(model_output, encoded_input['attention_mask'])
        embedding = F.normalize(embedding, p=2, dim=1).cpu().numpy()
    return embedding


def create_balanced_dataset(data_dict, model, tokenizer, device='cpu'):
    embeddings = {}
    for key, values in tqdm(data_dict.items()):
        embeddings[key] = {value: get_embedding(model, tokenizer, value, device) for value in values}

    queries, candidates, labels = [], [], []

    positive_pairs = []
    for values in data_dict.values():
        positive_pairs.extend(combinations(values, 2))

    hard_negative_pairs = []
    all_values = [(key, value, emb) for key, values in embeddings.items() for value, emb in values.items()]
    for key, values in tqdm(data_dict.items()):
        for value in values:
            emb1 = embeddings[key][value]
            distances = []
            for other_key, other_value, other_emb in all_values:
                if key != other_key:
                    dist = np.linalg.norm(emb1 - other_emb)
                    distances.append((dist, other_value))

            distances = sorted(distances, key=lambda x: x[0])
            hard_negative = distances[0][1]
            hard_negative_pairs.append((value, hard_negative))

    random.shuffle(hard_negative_pairs)
    hard_negative_pairs = hard_negative_pairs[:len(positive_pairs)]

    for value1, value2 in positive_pairs:
        queries.append(value1)
        candidates.append(value2)
        labels.append(1)

    for value1, value2 in hard_negative_pairs:
        queries.append(value1)
        candidates.append(value2)
        labels.append(0)

    df = pd.DataFrame({
        "query": queries,
        "candidate": candidates,
        "label": labels
    })
    return df


def initialize_qwen_model():
    model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def generate_solution(task_description, model, tokenizer, max_new_tokens=512):
    prompt = task_description
    messages = [
        {"role": "system",
         "content": "Your task is to generate python code for provided problem. Write ONLY code without any additional text comments."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=1.5
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.replace("```python", "").replace("```", "").strip()


class QueryCandidateDataset(Dataset):
    def __init__(self, queries, candidates, labels, tokenizer, max_length=512):
        self.queries = queries
        self.candidates = candidates
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        query = self.queries[idx]
        candidate = self.candidates[idx]
        label = self.labels[idx]

        inputs = self.tokenizer(
            query,
            candidate,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        inputs["labels"] = torch.tensor(label, dtype=torch.float)
        return inputs


def validate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
            labels = batch["labels"].to(device)

            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1)
            predictions = torch.sigmoid(logits)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    auc = roc_auc_score(all_labels, all_predictions)
    preds = (np.array(all_predictions) > 0.5).astype(int)
    f1 = f1_score(all_labels, preds)
    recall = recall_score(all_labels, preds)
    precision = precision_score(all_labels, preds)
    accuracy = accuracy_score(all_labels, preds)

    return auc, f1, recall, precision, accuracy


def train_model_with_validation(
        model, train_dataloader, val_dataloader, criterion,
        optimizer, device, num_epochs=3, metrics_file_path=None
):
    best_metrics = [0.0, 0.0, 0.0, 0.0, 0.0]
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
            inputs = {key: val.to(device) for key, val in batch.items()}

            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1)
            loss = criterion(logits, inputs["labels"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {epoch_loss / len(train_dataloader):.4f}")

        auc, f1, recall, precision, accuracy = validate_model(model, val_dataloader, device)
        if f1 > best_metrics[1]:
            best_metrics = [auc, f1, recall, precision, accuracy]

        if metrics_file_path:
            with open(metrics_file_path, "a") as f:
                f.write(f"""Epoch {epoch + 1}/{num_epochs} - Val AUC: {auc:.4f}, Val F1: {f1:.4f}
                        Val Accuracy: {accuracy:.4f} - Val Recall: {recall:.4f}, Val Precision: {precision:.4f}\n\n""")
        else:
            print(f"Epoch {epoch + 1}/{num_epochs} - Val AUC: {auc:.4f}, Val F1: {f1:.4f}")
            print(f"Val Accuracy: {accuracy:.4f} - Val Recall: {recall:.4f}, Val Precision: {precision:.4f}")

    return best_metrics


def launch_training(
        model_name, dataset_path="../data/cross_encoder_dataset_new.csv",
        saving_model_name=None, num_epochs=3, batch_size=32, lr=1e-5, metrics_file_path=None
):
    # "microsoft/codebert-base",
    # "mixedbread-ai/mxbai-rerank-base-v1",
    # "microsoft/graphcodebert-base",
    # Salesforce/codet5-large

    if metrics_file_path:
        # create file
        with open(metrics_file_path, "w") as f:
            f.write("Metrics\n")

    df = pd.read_csv(dataset_path)
    queries = df["query"].tolist()
    candidates = df["candidate"].tolist()
    labels = df["label"].tolist()

    train_queries, val_queries, train_candidates, val_candidates, train_labels, val_labels = train_test_split(
        queries, candidates, labels, test_size=0.2, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = QueryCandidateDataset(train_queries, train_candidates, train_labels, tokenizer)
    val_dataset = QueryCandidateDataset(val_queries, val_candidates, val_labels, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
    )
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr)

    best_metrics = train_model_with_validation(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        metrics_file_path=metrics_file_path,
    )

    if saving_model_name:
        model.save_pretrained(f"./{saving_model_name}")
        tokenizer.save_pretrained(f"./{saving_model_name}")

    return best_metrics


def launch_qwen():
    qwen_model, qwen_tokenizer = initialize_qwen_model()
    dataset = load_dataset("google-research-datasets/mbpp", "full")

    texts = []
    code_solutions = []

    for split in dataset:
        texts.extend(dataset[split]["text"])
        code_solutions.extend(dataset[split]["code"])

    task_descriptions = texts
    solutions_dict = {}
    num_solutions = 1
    max_new_tokens = 1024

    for i, task in enumerate(tqdm(task_descriptions)):
        solutions = []
        code = code_solutions[i]
        for _ in range(num_solutions):
            solution = generate_solution(task, qwen_model, qwen_tokenizer, max_new_tokens)
            solutions.append(solution)
        solutions.append(code)
        solutions_dict[task] = solutions

    with open("solutions_qwen.json", "w") as f:
        json.dump(solutions_dict, f, indent=4)

    model_name = "jinaai/jina-embeddings-v2-base-code"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    df = create_balanced_dataset(solutions_dict, model, tokenizer).sample(frac=1).reset_index(drop=True)
    df.to_csv("../data/cross_encoder_dataset.csv", index=False)
