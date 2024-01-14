from itertools import product
from collections import defaultdict

from Bio.Seq import Seq

def node_id(sample_id, contig_id):
    """Concatenates sample_id and contig_id to generate a node_id."""
    
    node_id = f"{sample_id}:{contig_id}"

    return node_id

def label_to_pair(label):
    """Convert a label into a pair of values based on predefined mappings."""
    
    # Define mappings of labels to pairs
    label_mappings = {
        "chromosome": [0, 1],
        "plasmid": [1, 0],
        "ambiguous": [1, 1],
        "unlabeled": [0, 0],
    }

    if label in label_mappings:
        return label_mappings[label]
    else:
        raise AssertionError(f"Unrecognized label: {label}")

def pair_to_label(pair):
    """Converts a pair of values into a label based on predefined mappings"""

    if pair == [0, 1]:
        return "chromosome"
    elif pair == [1, 0]:
        return "plasmid"
    elif pair == [1, 1]:
        return "ambiguous"
    elif pair == [0, 0]:
        return "unlabeled"
    else:
        raise AssertionError(f"Unrecognized pair: {pair}")

def prepare_kmer_lists(kmer_length):
    """
    Generates forward and reverse k-mers of given length.

    Args:
        kmer_length (int): Length of k-mers.

    Returns:
        tuple: A tuple containing lists of all k-mers and forward k-mers.
    """
    # Generate all possible k-mers
    kmers = ["".join(x) for x in product("ACGT", repeat=kmer_length)]

    # Initialize lists and sets for forward and reverse k-mers
    forward_kmers = []
    forward_kmer_set = set()
    reverse_kmer_set = set()

    # Iterate through all k-mers
    for kmer in kmers:
        if kmer not in forward_kmer_set and kmer not in reverse_kmer_set:
            # Add unique forward k-mers to the list and sets
            forward_kmers.append(kmer)
            forward_kmer_set.add(kmer)
            reverse_kmer_set.add(str(Seq(kmer).reverse_complement()))

    return kmers, forward_kmers

def calculate_kmer_distribution(sequence, kmer_length=5, scale=False):
    """Calculate k-mer distribution from a sequence"""
    
    assert kmer_length % 2 == 1, "K-mer length should be odd."

    # Generating k-mers
    k_mers, fwd_kmers = prepare_kmer_lists(kmer_length)

    # Using defaultdict for pseudocounts
    dict_kmer_count = defaultdict(lambda: 0.01)

    # Counting k-mers in the sequence
    for i in range(len(sequence) - kmer_length + 1):
        kmer = sequence[i:i + kmer_length]
        if kmer in dict_kmer_count:
            dict_kmer_count[kmer] += 1

    # Calculating k-mer counts
    k_mer_counts = [
        dict_kmer_count[k_mer] + dict_kmer_count[str(Seq(k_mer).reverse_complement())]
        for k_mer in fwd_kmers
    ]

    # Scaling k-mer counts if specified
    if scale:
        total_count = sum(k_mer_counts)
        k_mer_counts = [count / total_count for count in k_mer_counts]

    return k_mer_counts

def calculate_gc_content(sequence):
    """Calculate the GC content of a given DNA sequence."""

    # Count the number of 'G' and 'C' bases in the sequence
    gc_count = sum(base in 'GC' for base in sequence)

    # Count the total number of valid bases (A, C, G, T) in the sequence
    total_bases = sum(base in 'ACGT' for base in sequence)

    # Calculate GC content only if there are valid bases in the sequence
    if total_bases > 0:
        gc_content = round(gc_count / total_bases, 4)
    else:
        # Default GC content when there are no valid bases
        gc_content = 0.5

    return gc_content