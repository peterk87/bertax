from itertools import product
from random import randint
from typing import List, Optional, Tuple, Dict, Iterator, IO

from pathlib import Path


import keras
import keras_bert
import numpy as np

from bertax.const import CLASS_LABELS

ALPHABET = "ACGT"


def get_file_handle(path: Path) -> IO:
    if path.suffix == '.gz':
        import gzip
        return gzip.open(path, mode='rt')
    else:
        return open(path)


def parse_fastx(path: Path, handle: IO) -> Iterator[Tuple[str, str]]:
    if path.suffix == '.gz':
        suffix = path.suffixes[:-1][-1].lower()
    else:
        suffix = path.suffix.lower()
    if suffix in ['.fasta', '.fna', '.fa']:
        return parse_fasta(handle)
    elif suffix in ['.fastq', '.fq']:
        return parse_fastq(handle)
    else:
        raise NotImplementedError(f'No support for parsing files with extension "{suffix}"')


def parse_fastq(handle: IO) -> Iterator[Tuple[str, str]]:
    """Iterate over FASTQ records as string tuples"""
    header = ''
    seq = ''
    skip = False
    for line in handle:
        if skip:
            skip = False
            continue
        line = line.strip()
        if line == '':
            continue
        if line[0] == '@':
            header = line.replace('@', '').split(maxsplit=1)[0]
        elif line[0] == '+':
            yield header, seq
            skip = True
        else:
            seq = line.upper()


def parse_fasta(handle: IO) -> Iterator[Tuple[str, str]]:
    """Iterate over Fasta records as string tuples.
    Arguments:
     - handle - input stream opened in text mode
    For each record a tuple of two strings is returned, the FASTA title
    line (without the leading '>' character), and the sequence (with any
    whitespace removed). The title line is not divided up into an
    identifier (the first word) and comment or description.

    Source: https://github.com/biopython/biopython/blob/7753efc9fb4a08b845d3f41b3fc790c89743e196/Bio/SeqIO/FastaIO.py#L24
    """
    # Skip any text before the first record (e.g. blank lines, comments)
    for line in handle:
        if line[0] == ">":
            title = line[1:].rstrip()
            break
    else:
        # no break encountered - probably an empty file
        return

    # Main logic
    # Note, remove trailing whitespace, and any internal spaces
    # (and any embedded \r which are possible in mangled files
    # when not opened in universal read lines mode)
    lines = []
    for line in handle:
        if line[0] == ">":
            yield title, "".join(lines).replace(" ", "").replace("\r", "")
            lines = []
            title = line[1:].rstrip()
            continue
        lines.append(line.rstrip())

    yield title, "".join(lines).replace(" ", "").replace("\r", "")


def seq2kmers(seq: str, k: int = 3, stride: int = 3, pad: bool = True, to_upper: bool = True) -> List[str]:
    """transforms sequence to k-mer sequence.
    If specified, end will be padded so no character is lost"""
    if len(seq) < k:
        return [seq.ljust(k, "N")] if pad else []
    kmers = []
    i = 0
    for i in range(0, len(seq) - k + 1, stride):
        kmer = seq[i: i + k]
        if to_upper:
            kmers.append(kmer.upper())
        else:
            kmers.append(kmer)
    if (pad and len(seq) - (i + k)) % k != 0:
        kmers.append(seq[i + k:].ljust(k, "N"))
    return kmers


def seq2tokens(
        seq: str,
        token_dict: Dict[str, int],
        seq_length: int = 250,
        max_length: int = None,
        k: int = 3,
        stride: int = 3,
        window: bool = True,
        seq_len_like: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform raw sequence into list of tokens to be used for fine-tuning BERT

    Args:
        seq: nucleotide sequence
        token_dict: Keras BERT base token dict with nucleotide words
        seq_length:
        max_length:
        k:
        stride:
        window:
        seq_len_like:

    Returns:
        Tuple of: [0] indices int array; [1] zeroed array of length `max_length` for segments
    """
    if max_length is None:
        max_length = seq_length
    if seq_len_like is not None:
        seq_length = min(max_length, np.random.choice(seq_len_like))
    kmers = seq2kmers(seq, k=k, stride=stride, pad=True)
    if window:
        start = randint(0, max(len(kmers) - seq_length - 1, 0))
        end = start + seq_length - 1
    else:
        start = 0
        end = seq_length
    indices: List[int] = [token_dict["[CLS]"]] + [
        token_dict[word] if word in token_dict else token_dict["[UNK]"]
        for word in kmers[start:end]
    ]
    if len(indices) < max_length:
        indices += [token_dict[""]] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    segments = np.zeros(max_length, dtype=int)
    return np.array(indices), segments


def seq_frames(seq: str, frame_len: int, running_window=False, stride=1) -> Tuple[List[str], List[Tuple[int, int]]]:
    """returns all windows of seq with a maximum length of `frame_len` and specified stride
    if `running_window` else `frame_len` -- alongside the chunks' positions"""
    iterator = (
        range(0, len(seq) - frame_len + 1, stride)
        if running_window
        else range(0, len(seq), frame_len)
    )
    return [seq[i: i + frame_len] for i in iterator], [
        (i, i + frame_len) for i in iterator
    ]


def get_token_dict(alph: str = ALPHABET, k: int = 3) -> Dict[str, int]:
    """get token dictionary dict generated from `alph` and `k`"""
    token_dict = keras_bert.get_base_dict()
    for x in product(alph, repeat=k):
        word = ''.join(x)
        token_dict[word] = len(token_dict)
    return token_dict


def process_bert_tokens_batch(batch_x):
    """when `seq2tokens` is used as `custom_encode_sequence`, batches
    are generated as [[input1, input2], [input1, input2], ...]. In
    order to train, they have to be transformed to [input1s,
    input2s] with this function"""
    return np.array([x[0] for x in batch_x]), np.array([x[1] for x in batch_x])


def load_bert(bert_path, compile=False):
    """get bert model from path"""
    custom_objects = {
        "GlorotNormal": keras.initializers.glorot_normal,
        "GlorotUniform": keras.initializers.glorot_uniform,
    }
    custom_objects.update(keras_bert.get_custom_objects())
    return keras.models.load_model(
        bert_path, compile=compile, custom_objects=custom_objects
    )


def annotate_predictions(
        preds: List[np.ndarray],
        overwrite_class_labels: Optional[dict] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """annotates list of prediction arrays with provided or preset labels"""
    class_labels = (
        overwrite_class_labels if overwrite_class_labels is not None else CLASS_LABELS
    )
    return {
        rank: {label: v.astype(float) for label, v in zip(class_labels[rank], p.transpose())}
        for rank, p in zip(class_labels, preds)
    }


def best_predictions(
        preds_annotated: Dict[str, Dict[str, np.ndarray]]
) -> List[Tuple[str, str, np.ndarray]]:
    """returns best prediction classes alongside predicton confidences"""
    result = []
    for rank, label_prediction in preds_annotated.items():
        top_label = max(label_prediction, key=lambda label: label_prediction[label])
        result.append((rank, top_label, label_prediction[top_label]))
    return result
