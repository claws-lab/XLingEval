GPT_MODEL = "gpt-3.5-turbo"

TEMPERATURES = [0.0, 0.25, 0.5, 0.75, 1.0]
LANGUAGES = ["English", "Spanish", "Chinese", "Hindi"]
LANGUAGE_CODES = ["en", "es", "zh", "hi"]

HOME_DIR_LINUX = "/home/ahren"
HOME_DIR_LINUX_SERVER = "/nethome/yjin328"

# Features
ID = "id"
OPTION = "option"
QUESTION = "question"
QTEXT = "qtext"
QUESTION_TRANSLATED = "question_translated"
ANSWER = "answer"
ANSWER_TRANSLATED = "answer_translated"
ERROR = "error"
LABEL = "label"
LANGUAGE = "language"
TEMPERATURE = "temperature"
PRED = "pred"
PRED_BINARY = "pred_binary"

# Label
POSITIVE = "positive"
NEGATIVE = "negative"

LABEL2ID = {
    POSITIVE: 1,
    NEGATIVE: 0,
}

DATASET2LENGTH = {
    "healthqa": 1134,
    "liveqa": 245,
    "medicationqa": 690,
}

CONSISTENCY_METRICS_SIMILARITY = ["bert_sim",
                                  "bertscore_P",
                                  "bertscore_R", "bertscore_F1", "unigram_jaccard",
                                  "bigram_jaccard", "length_mean", "length_std"]

METRIC_NAME2FULLNAME = {
    "macro_precision": "Macro Precision",
    "macro_recall": "Macro Recall",
    "macro_f1": "Macro F1",
    "accuracy": "Accuracy",
    "auc": "AUC",
    'bert_sim': r"$\mathrm{sim}_{\mathrm{sent}}$",
    'bertscore_P': "BERTScore (Precision)",
    'bertscore_R': "BERTScore (Recall)",
    'bertscore_F1': r"$\mathrm{BERTScore}$",
    'unigram_jaccard': r"$\mathrm{sim}_{\mathrm{1-gram}}$",
    'bigram_jaccard': r"$\mathrm{sim}_{\mathrm{2-gram}}$",
    'length_mean': "Length",
    'length_std': "Std. of Length",
    'hdp_mean': r"$\mathrm{sim}_{\mathrm{HDP}}$",  # "Avg. Topical Similarity (HDP)",
    'lda20_mean': r"$\mathrm{sim}_{\mathrm{LDA}}^{20}$",  # "Avg. Topical Similarity (LDA w/ 20 Topics)",
    'hdp_std': "Std. Topical Similarity (HDP)",
}

VERIFIABILITY_METRICS_VISUALIZATION = ["macro_precision", "macro_recall", "macro_f1",
                                       "accuracy", "auc"]

CONSISTENCY_METRICS_TOPIC_MODELING = ["hdp_mean", "hdp_std"]
for num_topics in [10, 20, 50, 100, 200, 500]:
    CONSISTENCY_METRICS_TOPIC_MODELING += [f"lda{num_topics}_mean",
                                           f"lda{num_topics}_std"]

LANG2SHORT = {
    "English": "en",
    "Chinese": "zh-cn",
    "Hindi": "hi-in",
    "Spanish": "es",
}

MNLI_LABEL2ID = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
}

# Tasks
TRANSLATE = "translate"
MEDICAL = "medical"
PARAPHRASE = "paraphrase"

# split
TRAIN = "train"
DEV = "dev"
TEST = "test"



CHINESE_HINDI_PUNCTUATION = "，。？！；：।॥"

CONFIDENCE_LIKERT_SCALE = """
Confidence of your evaluation: 
Very confident (5): I have checked all aspects of the input sentences thoroughly. I am absolutely certain of my evaluation.
Quite confident (4): I am quite confident about my evaluation. It is unlikely, though possible, that I missed some elements that could otherwise have impacted my evaluation.
Somewhat confident (3): I am moderately confident about my evaluation. There is a chance I missed some aspects.
Not very confident (2): I am not very confident about my evaluation. I am able to defend my evaluation, but it is quite likely that I missed or did not understand some key details of the inputs.
Not confident (1): My evaluation is an educated guess.
"""
