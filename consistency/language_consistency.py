import re
from collections import defaultdict

import langid


def split_multilingual_paragraph(paragraph):
    # A more comprehensive regular expression pattern:
    # - (?<=[.!?।。！？]) looks behind to ensure one of the sentence-end punctuations is present
    # - \s* matches zero or more whitespace characters
    pattern = r'(?<=[.!?।。！？])\s*'

    # Split the paragraph using the pattern
    sentences = re.split(pattern, paragraph)

    # Remove any empty strings that might result from the split operation
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def detect_primary_language(sentence):
    lang, _ = langid.classify(sentence)
    return lang


def compute_language_percentage(sentences):
    language_counts = defaultdict(int)
    total_sentences = 0

    for sentence in sentences:
        lang = detect_primary_language(sentence)

        # Map the detected language to one of the four main languages you mentioned
        if lang not in ['en', 'es', 'zh', 'hi']:
            # for simplicity, you might want to map minor languages to the nearest major one
            # or just skip them
            continue

        language_counts[lang] += 1
        total_sentences += 1

    percentages = {}
    for lang, count in language_counts.items():
        percentages[lang] = (count / total_sentences) * 100

    return percentages


if __name__ == "__main__":

    # Test the function
    paragraph = "Hello, world! This is an English sentence. ¿Cómo estás? Esta es una oración en español. 你好吗？这是一个中文的句子。आप कैसे हैं? यह एक हिंदी वाक्य है।"
    sentences = split_multilingual_paragraph(paragraph)
    for sentence in sentences:
        print(sentence)

    sentences = [
        "This is an English sentence.",
        "Esta es una oración en español.",
        "这是一个中文句子。",
        "यह हिंदी में एक वाक्य है।",
        "This sentence is mainly in English with some 中文 words."
    ]

    percentages = compute_language_percentage(sentences)
    for lang, perc in percentages.items():
        print(f"Percentage of {lang}: {perc:.2f}%")
