# pip install datasets transformers faiss-cpu sentence-transformers tqdm
# pip install torch bert-score

import os
import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import numpy as np
import json

from transformers import (
    RagTokenizer,
    RagRetriever,
    RagSequenceForGeneration,
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM
)
from bert_score import score as bert_score

# loading the dataset
full_dataset = load_dataset("MedRAG/textbooks", split="train")

# for testing: code to sample the dataset
def sample_part(dataset):
    grouped = {}
    for ex in dataset:
        title = ex['title']
        if title not in grouped:
            grouped[title] = []
        grouped[title].append(ex)

    sampled = []
    for title, examples in grouped.items():
        n = len(examples) # // 200
        sampled.extend(examples[:n])
    return Dataset.from_list(sampled)

sampled_dataset = sample_part(full_dataset)

processed_dataset = full_dataset.map(
    lambda x, i: {
        "title": f"{x['title']} Doc {i}",
        "text": x["contents"]
    },
    with_indices=True
)

# some error about too many columns?
rag_dataset = processed_dataset.remove_columns(set(processed_dataset.column_names) - {"title", "text"})

# save dataset for retriever
dataset_path = "/content/textbook_full_dataset"
index_path = os.path.join(dataset_path, "faiss_index")
rag_dataset.save_to_disk(dataset_path)

# embedding w dpr, instead of knn stuff
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

def embed_texts(batch):
    inputs = ctx_tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt", max_length=256)
    with torch.no_grad():
        embeddings = ctx_encoder(**inputs).pooler_output
    return {"embeddings": embeddings.cpu().numpy()}

rag_dataset = rag_dataset.map(embed_texts, batched=True, batch_size=16) # this takes a while. experiment w batch size?
rag_dataset.add_faiss_index(column="embeddings")

rag_dataset.get_index("embeddings").save(index_path)
rag_dataset.drop_index("embeddings")
rag_dataset.save_to_disk(dataset_path)

# something is wrong here
dataset_path = "./textbook_dataset"
index_path = os.path.join(dataset_path, "faiss_index")
print("Saving dataset and FAISS index...")
rag_dataset.get_index("embeddings").save(index_path)
rag_dataset.drop_index("embeddings")
rag_dataset.save_to_disk(dataset_path)
print(f"Dataset and index saved to {dataset_path}")

# load rag model
print("Loading RAG model and retriever...")
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq",
    index_name="custom",
    passages_path=dataset_path,
    index_path=index_path,
    use_dummy_dataset=False,
)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)

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
            score += 0.5  # partial credit for correct textbook
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

# faithfulness
nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")

def compute_faithfulness(answer, context):
    inputs = nli_tokenizer(context, answer, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
    entailment_prob = probs[0][2].item()  # entailment class index
    return entailment_prob

qa_data = [] #synthetic
with open("final.json") as f:
  qa_data = json.load(f)

# def mkPrompt(question, retrieved_contexts, examples, system_prompt):
#     context_str = "\n\n".join([ctx["text"] for ctx in retrieved_contexts])
#     examples_str = "\n\n".join([
#         f"Q: {ex['question']}\nA: {ex['answer']}" for ex in examples
#     ])
#     return f"{system_prompt}\n\n{examples_str}\n\nQ: {question}\nContext:\n{context_str}\nA:"

def mkPrompt(question, retrieved_contexts, examples, system_prompt):
    context_str = "\n\n".join([f"Context Passage {i+1}:\n{ctx['text']}" for i, ctx in enumerate(retrieved_contexts)])
    examples_str = "\n\n".join([
        f"Q: {ex['question']}\nChain-of-thought: {ex['cot']}\nA: {ex['answer']}" for ex in examples
    ])
    return f"""{system_prompt}

{examples_str}

Q: {question}
{context_str}
Chain-of-thought:"""


from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

# dpr question encoder
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

few_shot_examples = [

    {
        "question": "What causes type 1 diabetes?",
        "cot": "Type 1 diabetes is an autoimmune condition where the body’s immune system attacks and destroys the insulin-producing beta cells in the pancreas. Without insulin, the body can't regulate blood glucose levels.",
        "answer": "The immune system destroys insulin-producing cells in the pancreas."
    },
    {
        "question": "Why do antibiotics not work on viruses?",
        "cot": "Antibiotics target specific features of bacteria, such as cell walls or protein synthesis mechanisms. Viruses lack these structures and replicate inside host cells, making antibiotics ineffective.",
        "answer": "Viruses lack the structures antibiotics target, so they are ineffective."
    },
    {
        "question": "How does a vaccine provide immunity?",
        "cot": "Vaccines introduce a harmless form of a pathogen or its components, prompting the immune system to create memory cells. These memory cells enable a faster and stronger response if the pathogen is encountered again.",
        "answer": "They train the immune system to recognize and fight the pathogen in future encounters."
    },
    {
        "question": "What is the function of the mitochondria?",
        "cot": "The mitochondria are organelles that generate most of the chemical energy needed to power the cell. This energy is produced in the form of ATP through a process called cellular respiration.",
        "answer": "The mitochondria produce ATP through cellular respiration."
    },
    {
        "question": "How do beta-blockers help in hypertension?",
        "cot": "Beta-blockers inhibit the effects of adrenaline on the heart. This slows the heartbeat and reduces blood pressure, making them effective for managing hypertension.",
        "answer": "They lower blood pressure by slowing the heart rate."
    },
   
]

# !huggingface-cli login
model_name = "meta-llama/Llama-2-7b-chat-hf"  # make sure you're approved
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # just for testing

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

results = []

for qa in tqdm(qa_data): # small subset: for qa in tqdm(qa_data[:5])
  #test_qa = qa_data[0]
  question = qa["question"]
  reference = qa["answer"]
  gold_doc_ids = qa.get("doc_ids", [])  # gold doc IDs

  # dpr retrieval
  q_tokens = question_tokenizer(question, return_tensors="pt")
  with torch.no_grad():
      q_embed = question_encoder(**q_tokens).pooler_output

  retrieved = retriever(
      question_input_ids=q_tokens["input_ids"],
      question_hidden_states=q_embed.cpu().numpy(),
      return_tensors="pt"
  )

  retrieved_doc_ids = retrieved["doc_ids"][0].tolist()
  retrieved_contexts = [{"text": sampled_dataset[idx]["text"]} for idx in retrieved_doc_ids[:3]]
  top_examples = few_shot_examples[:5]

  # build medprompt input
  prompt = mkPrompt(
      question=question,
      retrieved_contexts=retrieved_contexts,
      examples=top_examples,
      system_prompt="You are an expert medical professional. Given the context and examples, answer clearly."
  )

  print("PROMPT:", prompt)

  # run llama
  inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
  with torch.no_grad():
      outputs = model.generate(**inputs, max_new_tokens=300)
      generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

  print("generated answer: ", generated_answer)

  P, R, F1 = bert_score([generated_answer], [reference], lang="en", verbose=False)
  question_str = question  # or whatever input question is
  question_inputs = tokenizer(question_str, return_tensors="pt")

  # dpr question encoder
  question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
  question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
  with torch.no_grad():
      question_hidden = question_encoder(**question_inputs).pooler_output

  # top-k passages using retriever and dpr embedding
  retrieved = retriever(
      question_input_ids=question_inputs["input_ids"],
      question_hidden_states=question_hidden.cpu().numpy(), # must be njmpy
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

print(result)
