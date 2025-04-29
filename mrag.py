# pip install datasets transformers faiss-cpu sentence-transformers tqdm
# pip install datasets transformers torch sentence-transformers
# pip install bert-score transformers torch

from __future__ import annotations
import os, json, random, itertools, argparse, collections, tempfile, math
from datasets import load_dataset, Dataset, concatenate_datasets
from typing import List, Dict, Tuple, Any
import torch
from tqdm.auto import tqdm
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizerFast,
    RagTokenizer,
    RagRetriever,
    RagTokenForGeneration,
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    AutoModel,
    pipeline
)
from sentence_transformers import SentenceTransformer, util
import numpy as np
from bert_score import score
import nltk

# 1. Data Loading
def load_subset():
    dataset = load_dataset("MedRAG/textbooks", split='train')
    subset = []
    for title in set(dataset['title']):
        title_records = [row for row in dataset if row['title'] == title] #[:30]
        subset.extend(title_records)
    return subset

textbook_subset = load_subset()

class MultiHeadRetriever:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.model.eval()

    def get_multihead_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        all_head_embeddings = torch.stack(outputs.attentions).mean(dim=(0,1))
        return all_head_embeddings.cpu().numpy()

retriever = MultiHeadRetriever()
textbook_embeddings = [retriever.get_multihead_embeddings(doc['content']) for doc in textbook_subset]

def multihead_retrieve(query, k=5):
    query_embed = retriever.get_multihead_embeddings(query)
    query_flat = query_embed.reshape(-1)

    similarities = []
    for doc_embed in textbook_embeddings:
        doc_flat = doc_embed.reshape(-1)
        min_dim = min(query_flat.shape[0], doc_flat.shape[0])
        sim = np.dot(query_flat[:min_dim], doc_flat[:min_dim])
        similarities.append(sim)

    top_indices = np.argsort(similarities)[-k:][::-1]
    return [textbook_subset[i] for i in top_indices]

with open('final.json') as f:
    eval_data = json.load(f)

# !huggingface-cli login
model_name = "meta-llama/Llama-2-7b-chat-hf"

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
llama_tokenizer = AutoTokenizer.from_pretrained(model_name)
llama_model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map="auto",
)

def generate_llama_answer(context, question):
    tokenized_input = llama_tokenizer.encode(
        f"Context: {context}\nQuestion: {question}\n",
        return_tensors="pt",
        truncation=True,
        max_length=2048 
    ).to(llama_model.device)

    with torch.no_grad():
        generated_ids = llama_model.generate( # experiment with these parameters
            tokenized_input,
            max_new_tokens=256,
            min_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )
    answer_text = llama_tokenizer.decode(
        generated_ids[0][tokenized_input.shape[1]:], 
        skip_special_tokens=True
    )
    return answer_text.strip()

for qa in eval_data[:3]:  # subset
    print("Question:", qa['question'])
    retrieved = multihead_retrieve(qa['question'])
    context = "\n".join([doc['content'] for doc in retrieved])
    generated_answer = generate_llama_answer(context, qa['question'])
    print("Expected:", qa['answer'])
    print("Generated:", generated_answer)

try: # this punkt thing is not working
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

nltk.download('punkt')

# metrics FINALLY
 
# faithfulness 
nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")

def berts(generated, reference):
    _, _, F1 = score([generated], [reference], lang='en', verbose=False, rescale_with_baseline=True)
    return F1.item()

def faithful(answer, context):
    inputs = nli_tokenizer(context, answer, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
    return probs[0][2].item()  # index 2 is entailment

def prec(gold_doc_ids, retrieved_doc_ids):
    def get_group(doc_id): return "_".join(doc_id.split("_")[:-1])
    gold_groups = {get_group(did) for did in gold_doc_ids}
    score = 0.0
    for did in retrieved_doc_ids[:5]:
        if did in gold_doc_ids:
            score += 1.0
        elif get_group(did) in gold_groups:
            score += 0.5
    return score / 5

def mrr(gold_doc_ids, retrieved_doc_ids):
    def get_group(doc_id): return "_".join(doc_id.split("_")[:-1])
    gold_groups = {get_group(did) for did in gold_doc_ids}

    for rank, did in enumerate(retrieved_doc_ids, 1):
        if did in gold_doc_ids:
            return 1.0 / rank
        elif get_group(did) in gold_groups:
            return 0.5 / rank
    return 0.0

results = []

for qa in eval_data: #for qa in eval_data[:2]
    retrieved_docs = multihead_retrieve(qa["question"])
    context = "\n".join([d["content"] for d in retrieved_docs])
    generated_answer = generate_llama_answer(context, qa["question"])
    retrieved_ids = [d["id"] for d in retrieved_docs]
    gold_ids = qa.get("source", [])

    metrics = {
        "bertscore": berts(generated_answer, qa["answer"]),
        "faithfulness": faithful(generated_answer, context),
        "precision@5": prec(gold_ids, retrieved_ids),
        "mrr": mrr(gold_ids, retrieved_ids)
    }

    results.append({
        "question": qa["question"],
        **metrics
    })

avg_bertscore = np.mean([r['bertscore'] for r in results])
avg_faithfulness = np.mean([r['faithfulness'] for r in results])
avg_precision = np.mean([r['precision@5'] for r in results])
avg_mrr = np.mean([r['mrr'] for r in results])

print(f"BERTScore F1: {avg_bertscore:.4f}")
print(f"Faithfulness: {avg_faithfulness:.4f}")
print(f"Precision@5: {avg_precision:.4f}")
print(f"Mean Reciprocal Rank: {avg_mrr:.4f}")
