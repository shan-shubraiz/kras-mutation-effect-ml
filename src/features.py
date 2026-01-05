# src/features.py
from Bio import SeqIO
import numpy as np
import pandas as pd

def load_kras_sequence(fasta_path="data/kras.fasta"):
    record = next(SeqIO.parse(fasta_path, "fasta"))
    return str(record.seq)


# Kyte-Doolittle hydrophobicity
HYDRO = {
    'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,
    'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
    'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2
}
# Charge at physiological pH (approx.)
CHARGE = {'D':-1,'E':-1,'K':1,'R':1,'H':0.1}
# Polarity (binary)
POLAR = set(['R','N','D','Q','E','K','H','S','T','Y','C'])
# Approximate volume (A^3), small table
VOLUME = {
    'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,'G':60.1,
    'H':153.2,'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,'P':112.7,'S':89.0,
    'T':116.1,'W':227.8,'Y':193.6,'V':140.0
}
# Grantham distances (subset)
GRANTHAM = {
    ('G','D'):94,('G','V'):109,('G','A'):60,('Q','H'):24,('A','T'):58,('K','N'):94,
    # fallback default if not listed
}
# BLOSUM62 matrix (subset + fallback)
BLOSUM62 = {
    ('G','D'):-1,('G','V'):-3,('G','A'):0,('Q','H'):0,('A','T'):-1,('K','N'):-1,
}

AA = list(HYDRO.keys())
AA_IDX = {a:i for i,a in enumerate(AA)}

def blosum62(a,b):
    return BLOSUM62.get((a,b), BLOSUM62.get((b,a), -1))

def grantham(a,b):
    return GRANTHAM.get((a,b), GRANTHAM.get((b,a), 100))

def one_hot_window(seq, pos, k=2):
    start = max(0, pos-1-k)
    end = min(len(seq), pos-1+k+1)
    window = seq[start:end]
    vec = np.zeros(len(AA)*(2*k+1))
    for i,ch in enumerate(window):
        vec[i*len(AA)+AA_IDX[ch]] = 1
    return vec

def aa_prop(a):
    return {
        'hydro': HYDRO[a],
        'charge': CHARGE.get(a, 0),
        'polar': 1 if a in POLAR else 0,
        'volume': VOLUME[a]
    }

def mutation_features(seq, pos, ref, alt):
    # sanity
    assert seq[pos-1] == ref, f"Reference AA mismatch at pos {pos}"
    ref_prop = aa_prop(ref)
    alt_prop = aa_prop(alt)
    diff = {f'delta_{k}': alt_prop[k]-ref_prop[k] for k in ref_prop}

    feats = {
        'position': pos,
        'pos_norm': pos/len(seq),
        'blosum62': blosum62(ref, alt),
        'grantham': grantham(ref, alt),
        'ref_hydro': ref_prop['hydro'],
        'alt_hydro': alt_prop['hydro'],
        'ref_volume': ref_prop['volume'],
        'alt_volume': alt_prop['volume'],
        'ref_polar': ref_prop['polar'],
        'alt_polar': alt_prop['polar'],
        'ref_charge': ref_prop['charge'],
        'alt_charge': alt_prop['charge'],
        **diff
    }
    # local window one-hot
    window_vec = one_hot_window(seq, pos, k=2)
    # return dict + window as separate array
    return feats, window_vec


def load_mutations(csv_path="data/mutations.csv"):
    df = pd.read_csv(csv_path)
    return df

def featurize_dataset(csv_path="data/mutations.csv", fasta_path="data/kras.fasta"):
    seq = load_kras_sequence(fasta_path)
    df = load_mutations(csv_path)

    X_basic = []
    X_window = []
    y = []
    meta = []

    for _,row in df.iterrows():
        pos = int(row['position'])
        ref = row['ref']
        alt = row['alt']
        label = 1 if row['label']=='deleterious' else 0
        feats, window_vec = mutation_features(seq, pos, ref, alt)

        X_basic.append(feats)
        X_window.append(window_vec)
        y.append(label)
        meta.append(row['mutation'])

    X_basic_df = pd.DataFrame(X_basic)
    X_window_mat = np.vstack(X_window)
    y = np.array(y)
    return X_basic_df, X_window_mat, y, meta
