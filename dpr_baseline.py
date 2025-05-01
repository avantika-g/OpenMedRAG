# pip install datasets faiss-cpu --no-cache bert-score

import os
import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    RagTokenizer,
    RagRetriever,
    RagSequenceForGeneration,
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from bert_score import score as bert_score
import json

full_dataset = load_dataset("MedRAG/textbooks", split="train")

def partial_dataset(dataset):
    grouped = {}
    for ex in dataset:
        title = ex['title']
        if title not in grouped:
            grouped[title] = []
        grouped[title].append(ex)

    sampled = []
    for title, examples in grouped.items():
        n = len(examples) # // 200 - n is how many u want
        sampled.extend(examples[:n])
    return Dataset.from_list(sampled)

sampled_dataset = partial_dataset(full_dataset)

sampled_dataset = sampled_dataset.map(
    lambda x, i: {
        "title": f"{x['title']} Doc {i}",
        "text": x["contents"]
    },
    with_indices=True
)

rag_dataset = sampled_dataset.remove_columns(set(sampled_dataset.column_names) - {"title", "text"})

dataset_path = "/content/textbook_full_dataset"
index_path = os.path.join(dataset_path, "faiss_index")

ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

def embed_texts(batch):
    inputs = ctx_tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt", max_length=256)
    with torch.no_grad():
        embeddings = ctx_encoder(**inputs).pooler_output
    return {"embeddings": embeddings.cpu().numpy()}

rag_dataset = rag_dataset.map(embed_texts, batched=True, batch_size=16)
rag_dataset.add_faiss_index(column="embeddings")

rag_dataset.get_index("embeddings").save(index_path)
rag_dataset.drop_index("embeddings")

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq",
    index_name="custom",
    passages_path=dataset_path,
    index_path=index_path,
    use_dummy_dataset=False,
)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)

qa_data = []
with open("final.json") as f:
  qa_data = json.load(f)

nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")

# faithfulness
def compute_faithfulness(answer, context):
    inputs = nli_tokenizer(context, answer, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
    entailment_prob = probs[0][2].item() # entailment class index
    return entailment_prob

# metrics
def get_textbook_group(doc_id):
    return "_".join(doc_id.split("_")[:-1]) if "_" in doc_id else doc_id

def precision_k(gold_doc_ids, retrieved_doc_ids, k=5):
    gold_groups = {get_textbook_group(doc_id) for doc_id in gold_doc_ids}
    score = 0.0

    for doc_id in retrieved_doc_ids[:k]:
        if doc_id in gold_doc_ids:
            score += 1.0
        elif get_textbook_group(doc_id) in gold_groups:
            score += 0.5  # partial credit
    return score / k

def mrr(gold_doc_ids, retrieved_doc_ids):
    gold_groups = {get_textbook_group(doc_id) for doc_id in gold_doc_ids}

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold_doc_ids:
            return 1.0 / rank
        elif get_textbook_group(doc_id) in gold_groups:
            return 0.5 / rank  # partial reciprocal credit
    return 0.0

index_to_doc_id = [example["id"] for example in sampled_dataset]

from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

results = []

for qa in tqdm(qa_data):
  question = qa["question"]
  reference = qa["answer"]
  gold_doc_ids = qa.get("doc_id", [])  # gold doc IDs

  inputs = tokenizer(question, return_tensors="pt")
  with torch.no_grad():
      generated_ids = model.generate(
          input_ids=inputs["input_ids"],
          attention_mask=inputs["attention_mask"],
          num_return_sequences=1,
          num_beams=4,
      )
  generated_answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

  P, R, F1 = bert_score([generated_answer], [reference], lang="en", verbose=False)
  question_str = question  # or whatever your input question is
  question_inputs = tokenizer(question_str, return_tensors="pt")

  question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
  question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
  with torch.no_grad():
      question_hidden = question_encoder(**question_inputs).pooler_output

  # top-k passages
  retrieved = retriever(
      question_input_ids=question_inputs["input_ids"],
      question_hidden_states=question_hidden.cpu().numpy(), 
      return_tensors="pt"
  )

  retrieved_doc_ids = retrieved["doc_ids"][0].tolist()
  top_doc_index = retrieved_doc_ids[0]

  top_doc_id = index_to_doc_id[top_doc_index]
  top_context = sampled_dataset[top_doc_index]["text"]

  faithfulness = compute_faithfulness(generated_answer, top_context)
  doc_ids_converted = [index_to_doc_id[idx] for idx in retrieved_doc_ids]
  p_at_5 = precision_k(gold_doc_ids, doc_ids_converted, k=5)
  rr = mrr(gold_doc_ids, doc_ids_converted)

  result = {
      "question": question,
      "reference": reference,
      "generated": generated_answer,
      "bertscore_f1": round(F1.item(), 4),
      "faithfulness_entailment_prob": round(faithfulness, 4),
      "precision@5": round(p_at_5, 4),
      "reciprocal_rank": round(rr, 4),
  }
  print(result)
  results.append(result)

