import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, AutoModelForSequenceClassification, AutoTokenizer, RagRetriever, RagTokenizer, RagSequenceForGeneration
from datasets import load_dataset, Dataset as HFDataset
import json
from tqdm import tqdm
import random
from collections import defaultdict
from bert_score import score as bert_score
import os

class GetTriples(Dataset):
    def __init__(self, triples, src_to_text, question_tokenizer, context_tokenizer, max_length=256):
        self.triples = triples
        self.src_to_text = src_to_text
        self.question_tokenizer = question_tokenizer
        self.context_tokenizer = context_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        example = self.triples[idx]
        question = example['question']
        pos_source = example['positive_source']
        neg_source = example['negative_source']

        pos_pass = self.src_to_text.get(pos_source, "")
        neg_pass = self.src_to_text.get(neg_source, "")

        question_inputs = self.question_tokenizer(question, return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_length)
        pos_inputs = self.context_tokenizer(pos_pass, return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_length)
        neg_inputs = self.context_tokenizer(neg_pass, return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_length)

        return {
            "question_input_ids": question_inputs["input_ids"].squeeze(0),
            "question_attention_mask": question_inputs["attention_mask"].squeeze(0),
            "pos_input_ids": pos_inputs["input_ids"].squeeze(0),
            "pos_attention_mask": pos_inputs["attention_mask"].squeeze(0),
            "neg_input_ids": neg_inputs["input_ids"].squeeze(0),
            "neg_attention_mask": neg_inputs["attention_mask"].squeeze(0),
        }

# Create triples using the synthetic dataset (final.json),
# by adding random negatives

with open("synthetic_data.json", "r") as f:
    qa_data = json.load(f)

map = defaultdict(list)

for example in qa_data:
    source = example["doc_id"]
    parts = source.split("_")
    author = "_".join(parts[:-1])
    passage_num = int(parts[-1])

    map[author].append(passage_num)

# add random negatives
new_qa_data = []

for example in qa_data:
    question = example["question"]
    answer = example["answer"]
    source = example["doc_id"]

    parts = source.split("_")
    author = "_".join(parts[:-1])
    passage_num = int(parts[-1])

    available_passages = map[author]
    neg_cands = [num for num in available_passages if num != passage_num]

    if not neg_cands:
        continue

    neg_num = random.choice(neg_cands)

    new_entry = {
        "question": question,
        "answer": answer,
        "positive_source": source,
        "negative_source": f"{author}_{neg_num}",
    }
    new_qa_data.append(new_entry)

with open("final_finetune.json", "w") as f:
    json.dump(new_qa_data, f, indent=2)

full_dataset = load_dataset("MedRAG/textbooks", split="train")

def get_data_by_titles(dataset):
    grouped = {}
    for example in dataset:
        title = example['title']
        if title not in grouped:
            grouped[title] = []
        grouped[title].append(example)

    sampled = []
    for title, examples in grouped.items():
        n = len(examples)
        sampled.extend(examples[:n])
    return HFDataset.from_list(sampled)

dataset_full_sample = get_data_by_titles(full_dataset)

src_to_text = {example['id']: example['contents'] for example in dataset_full_sample}

with open("final_finetune.json", "r") as f:
    qa_triples = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)

question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

dataset = GetTriples(qa_triples, src_to_text, question_tokenizer, context_tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.AdamW(list(question_encoder.parameters()) + list(context_encoder.parameters()), lr=2e-5)

for epoch in range(10):
    question_encoder.train()
    context_encoder.train()

    total_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()

        question_input_ids = batch["question_input_ids"].to(device)
        question_attention_mask = batch["question_attention_mask"].to(device)
        pos_input_ids = batch["pos_input_ids"].to(device)
        pos_attention_mask = batch["pos_attention_mask"].to(device)
        neg_input_ids = batch["neg_input_ids"].to(device)
        neg_attention_mask = batch["neg_attention_mask"].to(device)

        q_embed = question_encoder(input_ids=question_input_ids, attention_mask=question_attention_mask).pooler_output
        pos = context_encoder(input_ids=pos_input_ids, attention_mask=pos_attention_mask).pooler_output
        neg = context_encoder(input_ids=neg_input_ids, attention_mask=neg_attention_mask).pooler_output

        loss = torch.nn.functional.margin_ranking_loss(
            (q_embed * pos).sum(dim=1),
            (q_embed * neg).sum(dim=1),
            target=torch.ones_like((q_embed * pos).sum(dim=1)),
            margin=1.0,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

question_encoder.save_pretrained("fine_tuned_question_encoder")
context_encoder.save_pretrained("fine_tuned_context_encoder")

nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")

def faith_calc(answer, context):
    inputs = nli_tokenizer(context, answer, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
    entailment_prob = probs[0][2].item()
    return entailment_prob

def get_textbook_group(doc_id):
    if "_" in doc_id:
        return "_".join(doc_id.split("_")[:-1])
    else:
        return doc_id

def p_at_k(gold_doc_ids, retrieved_doc_ids, k=5):
    gold_groups = {get_textbook_group(doc_id) for doc_id in gold_doc_ids}
    score = 0.0

    for doc_id in retrieved_doc_ids[:k]:
        if doc_id in gold_doc_ids:
            score += 1.0
        elif get_textbook_group(doc_id) in gold_groups:
            score += 0.5  # partial credit
    return score / k

def mrr_calc(gold_doc_ids, retrieved_doc_ids):
    gold_groups = {get_textbook_group(doc_id) for doc_id in gold_doc_ids}

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold_doc_ids:
            return 1.0 / rank
        elif get_textbook_group(doc_id) in gold_groups:
            return 0.5 / rank  # partial credit
    return 0.0

index_to_doc_id = [example["id"] for example in dataset_full_sample]

# insert finetuned retrieval models into full pipeline
# taken from baseline code
def get_data_by_titles(dataset):
    grouped = {}
    for example in dataset:
        title = example['title']
        if title not in grouped:
            grouped[title] = []
        grouped[title].append(example)

    sampled = []
    for title, examples in grouped.items():
        n = len(examples)
        sampled.extend(examples[:n])
    return Dataset.from_list(sampled)

dataset_full_sample = get_data_by_titles(full_dataset)

# same as above -- maybe merge code?
dataset_full_sample = dataset_full_sample.map(
    lambda x, i: {
        "title": f"{x['title']} Doc {i}",
        "text": x["contents"]
    },
    with_indices=True
)

rag_dataset = dataset_full_sample.remove_columns(set(dataset_full_sample.column_names) - {"title", "text"})

dataset_path = "/content/textbook_full_dataset"
index_path = os.path.join(dataset_path, "faiss_index")

fine_tuned_context_encoder_path = "fine_tuned_context_encoder"
ctx_encoder = DPRContextEncoder.from_pretrained(fine_tuned_context_encoder_path)
ctx_tokenizer = context_tokenizer

def embed_texts(batch):
    inputs = ctx_tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt", max_length=256)
    with torch.no_grad():
        embeddings = ctx_encoder(**inputs).pooler_output
    return {"embeddings": embeddings.cpu().numpy()}

rag_dataset = rag_dataset.map(embed_texts, batched=True, batch_size=16)
rag_dataset.add_faiss_index(column="embeddings")
rag_dataset.get_index("embeddings").save(index_path)
rag_dataset.drop_index("embeddings")

fine_tuned_question_encoder_path = "fine_tuned_question_encoder"
question_encoder = DPRQuestionEncoder.from_pretrained(fine_tuned_question_encoder_path)
question_tokenizer = question_tokenizer

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq",
    index_name="custom",
    passages_path=dataset_path,
    index_path=index_path,
    use_dummy_dataset=False,
    question_encoder_name_or_path="fine_tuned_question_encoder",
    generator_name_or_path="facebook/bart-base"
)

model = RagSequenceForGeneration.from_pretrained(
    "facebook/rag-sequence-nq",
    retriever=retriever,
)

results = []

for qa in tqdm(qa_data):
    question = qa["question"]
    reference = qa["answer"]
    gold_doc_ids = qa.get("doc_id", [])
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

    question_inputs = tokenizer(question, return_tensors="pt")
    with torch.no_grad():
        question_hidden = question_encoder(**question_inputs).pooler_output

    retrieved = retriever(
        question_input_ids=question_inputs["input_ids"],
        question_hidden_states=question_hidden.cpu().numpy(),
        return_tensors="pt"
    )

    retrieved_doc_ids = retrieved["doc_ids"][0].tolist()
    top_doc_index = retrieved_doc_ids[0]
    top_context = dataset_full_sample[top_doc_index]["text"]

    doc_ids_converted = [index_to_doc_id[idx] for idx in retrieved_doc_ids]
    p_at_5 = p_at_k(gold_doc_ids, doc_ids_converted, k=5)
    rr = mrr_calc(gold_doc_ids, doc_ids_converted)

    result = {
        "question": question,
        "reference": reference,
        "generated": generated_answer,
        "bertscore_f1": round(F1.item(), 4),
        "faithfulness_entailment_prob": round(faith_calc(generated_answer, dataset_full_sample[top_doc_index]["text"]), 4),
        "precision@5": round(p_at_5, 4),
        "reciprocal_rank": round(rr, 4),
    }
    print(result)
    results.append(result)