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
    TrainingArguments,
    Trainer,
    AutoModel,
    pipeline
)
from sentence_transformers import SentenceTransformer, util
import numpy as np
from bert_score import score
import nltk

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

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # only for training
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

'''
finetuning
- contrastive learning
- bioBERT encoder
'''

# pip install datasets transformers 
from torch.utils.data import DataLoader, Dataset

textbook_data = load_dataset("MedRAG/textbooks", split='train')
docid_to_content = {doc['id']: doc['content'] for doc in textbook_data}

with open('synthetic_data.json') as f:
    synthetic_data = json.load(f)

class MRAGRetriever(torch.nn.Module):
    def __init__(self, model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                               return_tensors="pt", max_length=512)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS pooling

def contrastive_loss(question_emb, pos_emb, neg_emb, margin=0.3):
    pos_sim = torch.cosine_similarity(question_emb, pos_emb)
    neg_sim = torch.cosine_similarity(question_emb, neg_emb)
    return torch.relu(neg_sim - pos_sim + margin).mean()

retriever = MRAGRetriever()
optimizer = torch.optim.AdamW(retriever.parameters(), lr=2e-5) # experiment w this

for epoch in range(3):  # adjust based on how much data
    total_loss = 0
    for example in synthetic_data:
        question = [example["question"]]
        pos_ctx = [docid_to_content[example["positive_source"]]]
        neg_ctx = [docid_to_content[example["negative_source"]]]

        q_emb = retriever(question)
        p_emb = retriever(pos_ctx)
        n_emb = retriever(neg_ctx)

        loss = contrastive_loss(q_emb, p_emb, n_emb)

        # uodate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(synthetic_data):.4f}")

# to save model and train w previous eval loop
retriever.model.save_pretrained("./medrag_retriever")
retriever.tokenizer.save_pretrained("./medrag_retriever")

# need to update this slightly before eval loop
def multihead_retrieve(question, k=5):
    question_emb = retriever([question])
    question_emb = question_emb.detach()

    doc_embeddings = []
    doc_ids = []
    for doc_id, content in docid_to_content.items():
        emb = retriever([content]).detach()
        doc_embeddings.append(emb)
        doc_ids.append(doc_id)

    doc_embeddings = torch.stack(doc_embeddings).squeeze(1)  # [N, hidden_dim]
    sims = torch.cosine_similarity(question_emb, doc_embeddings)  # [N]
    topk_indices = torch.topk(sims, k).indices.tolist()

    return [{"id": doc_ids[i], "content": docid_to_content[doc_ids[i]]} for i in topk_indices]


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