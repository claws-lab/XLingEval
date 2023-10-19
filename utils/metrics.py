import math
from collections import Counter
from typing import List
import numpy as np
import torch
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score


def compute_unigram_probabilities(corpus: List[str]) -> dict:
    # Flatten the list of sentences in the corpus into a list of words
    word_list = [word for sentence in corpus for word in sentence.split()]

    word_list2 = []

    for sentence in corpus:
        word_list2 += word_tokenize(sentence)

    # Count the frequency of each word in the corpus
    word_counts = Counter(word_list)
    total_words = len(word_list)
    # Calculate the unigram probabilities
    unigram_probabilities = {word: count / total_words for word, count in
                             word_counts.items()}
    return unigram_probabilities


def compute_lexical_similarity(s1: str, s2: str, unigram_probabilities: dict):
    # Create sets of the words in the snippets
    s1_words = set(s1.split())
    s2_words = set(s2.split())

    # Compute the intersection (common words) and union (all words) of the two sets
    common_words = s1_words & s2_words

    # Consider unique words only
    all_words = s1_words | s2_words

    # Consider all words, including repeated ones
    # all_words = list(s1_words) + list(s2_words)

    # Compute the log probability weighted sum of the common words
    common_sum = sum(
        [math.log(unigram_probabilities[word]) for word in common_words if
         word in unigram_probabilities])

    # Compute the log probability weighted sum of all the words
    all_sum = sum(
        [math.log(unigram_probabilities[word]) for word in all_words if
         word in unigram_probabilities])

    # Compute the lexical similarity
    sim_l = (2 * common_sum) / all_sum if all_sum else 0

    return sim_l


def pairwise_cos_sim(A: torch.Tensor, B: torch.Tensor, device="cuda:0"):
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A).to(device)

    if isinstance(B, np.ndarray):
        B = torch.from_numpy(B).to(device)

    from torch.nn.functional import normalize
    A_norm = normalize(A, dim=1)  # Normalize the rows of A
    B_norm = normalize(B, dim=1)  # Normalize the rows of B
    cos_sim = torch.matmul(A_norm,
                           B_norm.t())  # Calculate the cosine similarity
    return cos_sim


def jaccard_sim(text1: str, text2: str, ngram: int = 1) -> float:
    # Create the CountVectorizer object. Note that `binary=True` so we only care about the presence of each word but not #times it appears
    vectorizer = CountVectorizer(ngram_range=(ngram, ngram), binary=True)

    # Create the document-term matrix
    dtm = vectorizer.fit_transform([text1, text2]).toarray()

    # Calculate the Jaccard similarity
    return jaccard_score(dtm[0], dtm[1])

from collections import Counter
import math


def compute_entropy_rate(text):
    words = text.split()
    word_counts = Counter(words)
    total_words = len(words)

    entropy = -sum([(count / total_words) * math.log2(count / total_words)
                    for count in word_counts.values()])

    # Normalize by the length of the text to get entropy rate
    return entropy / total_words


if __name__ == '__main__':
    paragraphs = ["This is an example text. This is another example text. This text is meant as an example.",
                  "Studies have shown that the development of language models and resources is disproportionately focused on English",
                  "SMILES, which stands for Simplified Molecular Input Line Entry System, is a widely used notation for representing the structure of chemical molecules. ",
                  "Souvlaki is a Greek dish typically made with grilled meat, vegetables, and seasonings skewered onto small wooden sticks."
                  ]

    for sentence in paragraphs:
        print(f"Entropy rate: {compute_entropy_rate(sentence)}")
