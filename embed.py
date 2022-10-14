import torch
import re
import sys
import os

from tqdm import tqdm
from Bio import SeqIO
from transformers import T5Tokenizer, T5EncoderModel

def preprocess_from_fasta(path):
    labels, seqs = [], []
    for record in SeqIO.parse(path, "fasta"):
        labels.append(record.id)

        seq = str(record.seq)[:1022]
        seq = re.sub(r'[JUZOB\*]', 'X', seq)
        seqs.append(' '.join(list(seq)))

    return labels, seqs

def save_embeddings(emb_dict, out_path):
    with h5py.File(str(out_path), "w") as hf:
        for sequence_id, embedding in emb_dict.items():
            # noinspection PyUnboundLocalVariable
            hf.create_dataset(sequence_id, data=embedding)
    return None


device = 'gpu' if torch.cuda.is_available() else 'cpu'
batch_size = 64

    #Load model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float16)


fasta_file = "/Users/frishman/Dropbox/Bioinformatics/projects/embed/seqs.fa"

seqs = {seq.id: re.sub("[UZOB]", "X", str(seq.seq)) for seq in SeqIO.parse(fasta_file, "fasta")}
seqs = sorted(seqs.items(), key=lambda kv: len(seqs[kv[0]]), reverse=True)

batch = []

results = {"residue_embs": dict(),
           "protein_embs": dict()}

print(f"Transfering model to {device}...")
model.to(device)

total_batches = math.ceil(len(seqs) / batch_size)

for batch_number, (head, seq) in enumerate(seqs):
    seq = ' '.join(list(seq))
    batch.append((head, seq, len(seq)))


print(batch_size, " ", len(batch), " ", batch_number, " ", len(seqs))
if batch_size == len(batch) or batch_number == len(seqs) -1:

    heads, seqs, seq_lens = zip(*batch)
    batch = []

    token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
    print(type(token_encoding))
    input_ids = torch.tensor(token_encoding['input_ids']).to(device)
    attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

    with torch.no_grad():
        print(f"Calculating {batch_number}/{total_batches}")
        embedding_rpr = model(input_ids=input_ids, attention_mask=attention_mask)

    for i, head in enumerate(heads):
        emb = embedding_rpr.last_hidden_state[i:seq_lens[i]]
        emb = emb.detach.cpu().numpy()
        results["residue_embs"][head] = emb.detach().cpu().numpy().squeeze()
        results["protein_embs"][head] = emb.mean(dim=0).detach().cpu().numpy().squeeze()

    save_embeddings(results["residue_embs"], os.path.join("/Users/frishman/Dropbox/Bioinformatics/projects/embed/residue_embs.h5"))
    save_embeddings(results["protein_embs"], os.path.join("/Users/frishman/Dropbox/Bioinformatics/projects/embed/protein_embs.h5"))

