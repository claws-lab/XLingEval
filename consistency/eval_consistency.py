import os.path as osp
import platform
import re
import sys
import time
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from bert_score import score, get_idf_dict, bert_cos_score_idf
from nltk import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

import const
from arguments import args
from setup import project_setup, openai_setup
from utils.metrics import pairwise_cos_sim, jaccard_sim
import warnings
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertModel*")

if platform.system() == "Linux" or platform.system() == "Windows":
    DEVICE = "cuda:0"

elif platform.system() == "Darwin":
    DEVICE = "mps:0"

else:
    raise NotImplementedError

# Get the English stopwords
stopwords_list = set(stopwords.words("english"))


def score(
    cands,
    refs,
    transformer_model,
    model_type,
    tokenizer,
    num_layers=None,
    verbose=False,
    idf=False,
    device=None,
    batch_size=64,
    nthreads=4,
    lang: str="en",
    all_layers=False,
    rescale_with_baseline=False,
    baseline_path=None,
):
    """
    BERTScore metric.

    Args:
        - :param: `cands` (list of str): candidate sentences
        - :param: `refs` (list of str or list of list of str): reference sentences
        - :param: `model_type` (str): bert specification, default using the suggested
                  model for the target langauge; has to specify at least one of
                  `model_type` or `lang`
        - :param: `num_layers` (int): the layer of representation to use.
                  default using the number of layer tuned on WMT16 correlation data
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `idf` (bool or dict): use idf weighting, can also be a precomputed idf_dict
        - :param: `device` (str): on which the contextual embedding model will be allocated on.
                  If this argument is None, the model lives on cuda:0 if cuda is available.
        - :param: `nthreads` (int): number of threads
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `lang` (str): language of the sentences; has to specify
                  at least one of `model_type` or `lang`. `lang` needs to be
                  specified when `rescale_with_baseline` is True.
        - :param: `return_hash` (bool): return hash code of the setting
        - :param: `rescale_with_baseline` (bool): rescale bertscore with pre-computed baseline
        - :param: `baseline_path` (str): customized baseline file

    Return:
        - :param: `(P, R, F)`: each is of shape (N); N = number of input
                  candidate reference pairs. if returning hashcode, the
                  output will be ((P, R, F), hashcode). If a candidate have
                  multiple references, the returned score of this candidate is
                  the *best* score among all references.
    """
    assert len(cands) == len(refs), "Different number of candidates and references"


    ref_group_boundaries = None
    if not isinstance(refs[0], str):
        ref_group_boundaries = []
        ori_cands, ori_refs = cands, refs
        cands, refs = [], []
        count = 0
        for cand, ref_group in zip(ori_cands, ori_refs):
            cands += [cand] * len(ref_group)
            refs += ref_group
            ref_group_boundaries.append((count, count + len(ref_group)))
            count += len(ref_group)


    if not idf:
        idf_dict = defaultdict(lambda: 1.0)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[tokenizer.sep_token_id] = 0
        idf_dict[tokenizer.cls_token_id] = 0
    elif isinstance(idf, dict):
        if verbose:
            print("using predefined IDF dict...")
        idf_dict = idf
    else:
        if verbose:
            print("preparing IDF dict...")
        start = time.perf_counter()
        idf_dict = get_idf_dict(refs, tokenizer, nthreads=nthreads)
        if verbose:
            print("done in {:.2f} seconds".format(time.perf_counter() - start))

    if verbose:
        print("calculating scores...")
    start = time.perf_counter()
    all_preds = bert_cos_score_idf(
        transformer_model,
        refs,
        cands,
        tokenizer,
        idf_dict,
        verbose=verbose,
        device=device,
        batch_size=batch_size,
        all_layers=all_layers,
    ).cpu()

    if ref_group_boundaries is not None:
        max_preds = []
        for beg, end in ref_group_boundaries:
            max_preds.append(all_preds[beg:end].max(dim=0)[0])
        all_preds = torch.stack(max_preds, dim=0)

    if rescale_with_baseline:
        if baseline_path is None:
            baseline_path = osp.join(osp.dirname(__file__), f"rescale_baseline/{lang}/{model_type}.tsv")
        if osp.isfile(baseline_path):
            if not all_layers:
                baselines = torch.from_numpy(pd.read_csv(baseline_path).iloc[num_layers].to_numpy())[1:].float()
            else:
                baselines = torch.from_numpy(pd.read_csv(baseline_path).to_numpy())[:, 1:].unsqueeze(1).float()

            all_preds = (all_preds - baselines) / (1 - baselines)
        else:
            print(
                f"Warning: Baseline not Found for {model_type} on {lang} at {baseline_path}", file=sys.stderr,
            )

    out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]  # P, R, F

    if verbose:
        time_diff = time.perf_counter() - start
        print(f"done in {time_diff:.2f} seconds, {len(refs) / time_diff:.2f} sentences/sec")

    return out



def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

def get_topic_mean_std(cos_sim):
    upper_tri_indices = torch.triu_indices(cos_sim.shape[1],
                                           cos_sim.shape[2], offset=1)

    # Use these indices to get the upper triangular part of the matrix.
    upper_tri = cos_sim[:, upper_tri_indices[0], upper_tri_indices[1]]

    # Calculate the average.
    mean = torch.mean(upper_tri, dim=1)
    std = torch.std(upper_tri, dim=1)

    return mean.cpu().numpy(), std.cpu().numpy()


def consistency(language: str):
    language_prefix = "" if args.target_language == "English" else "TRANSLATED_"
    model_prefix = "" if args.model == "gpt35" else f"{args.model}_"

    print(f"Prefix of this experiment: {language_prefix}{model_prefix}")

    if args.dataset_name in ["liveqa", "medicationqa"]:

        path = osp.join(args.output_dir, "consistency",
                        f"{language_prefix}{model_prefix}{args.dataset_name}_consistency_temp{args.temperature}_{language}.xlsx")

        path_results = osp.join(args.output_dir, "summary",
                                f"{model_prefix}{args.dataset_name}_consistency_temp{args.temperature}.xlsx")

        # path_topic_modeling = osp.join(args.output_dir, "summary",
        #                                f"{args.dataset_name}_topic_modeling_temp{args.temperature}.xlsx")

    elif args.dataset_name in ["healthqa"]:

        path = osp.join(args.output_dir, "consistency",
                        f"{language_prefix}{model_prefix}{args.dataset_name}_{args.split}_consistency_temp{args.temperature}"
                        f"_{language}.xlsx")

        path_results = osp.join(args.output_dir, "summary",
                                f"{model_prefix}{args.dataset_name}_consistency_{args.split}_temp{args.temperature}.xlsx")


    else:
        raise NotImplementedError

    if not osp.exists(path):
        print(f"Error: Not found ({path})")

        return
    df = pd.read_excel(path)

    if len(df) < 5:
        print(f"Error: too short ({path})")
        return

    # TODO
    if osp.exists(path_results):
        excel_file = pd.ExcelFile(path_results)

        if args.target_language in excel_file.sheet_names:
            results_df = pd.read_excel(path_results,
                                       sheet_name=args.target_language,
                                       engine="openpyxl")
            print(f"Loaded {len(results_df)} examples from {path}")

        else:
            results_df = pd.DataFrame()
            results_df[const.ID] = df[const.ID]


    else:
        results_df = pd.DataFrame(
            columns=[const.ID] + const.CONSISTENCY_METRICS_SIMILARITY + const.CONSISTENCY_METRICS_TOPIC_MODELING)
        results_df[const.ID] = df[const.ID]

    df.set_index(const.ID, inplace=True)

    results_df.set_index(const.ID, inplace=True)

    def save(df_to_save, path_results: str, sheet_name: str):

        if osp.exists(path_results):
            with pd.ExcelWriter(path_results, engine='openpyxl', mode='a',
                                     if_sheet_exists='replace') as writer:
                df_to_save.to_excel(writer, index=True,
                                    sheet_name=sheet_name)

        else:
            with pd.ExcelWriter(path_results, engine='openpyxl', mode='w') as writer:
                df_to_save.to_excel(writer, index=True,
                                    sheet_name=sheet_name)

    if args.do_similarity:
        if osp.exists(const.HOME_DIR_LINUX):
            local_dir = "../models/"
        else:
            local_dir = ""

        MAX_LENGTH = 512
        model_type = "bert-base-uncased"
        model = SentenceTransformer(f"{local_dir}{model_type}", device=DEVICE)
        model_path = f"{local_dir}{model_type}"
        tokenizer = AutoTokenizer.from_pretrained(model_path, max_length=MAX_LENGTH, truncation=True)
        transformer_model = AutoModel.from_pretrained(model_path, max_length=MAX_LENGTH)
        tokenizer.model_max_length = MAX_LENGTH

        transformer_model.to(DEVICE)
        model.to(DEVICE)

        # all-MiniLM-L6-v2 is a light weight model
        # model_type = 'all-MiniLM-L6-v2'

        if not 'bert_sim' in results_df.columns or results_df.loc[:, "bert_sim"].isna().all() or args.fill_null_values:
            print("Evaluating similarity metrics ...")

            # bert_model = AutoModel.from_pretrained(osp.join(args.model_dir, model_type))

            # tokenizer = AutoTokenizer.from_pretrained(osp.join(args.model_dir, model_type))
            P_d, R_d, F1_d = defaultdict(list), defaultdict(list), defaultdict(
                list)

            for idx_row in range(0, len(df) * args.interval,
                                 args.interval):
                row = df.loc[idx_row]
                # results_df.loc[idx_row, const.ID] = row[const.ID]

                # Assuming responses is a list of responses from the chatbot

                responses = [row[f"answer_{i}"] for i in
                             range(args.num_answers)]



                if any([(not isinstance(response, str)) for response in responses]):
                    print(f"\tNon-str answers: {row.name}")
                    continue

                responses = [re.sub(r"\s+", " ", response.replace("### Input:", "").replace("### Explanation:",
                                                                                            "").replace(
                    "### "
                                                                                                        "Response:",
                                                                                                        "").replace(
                    "### Instruction:", "").replace("#", " ")) for response in
                             responses]

                if any([(not any([c.isalpha() for c in response])) for response in
                        responses]):
                    print(f"\tAnswers without characters: {row.name}")
                    continue

                # tokenized_answers = [tokenizer.encode(answer, add_special_tokens=True) for answer in responses]

                responses_words = [word_tokenize(response) for response in
                                   responses]

                # Compute Semantic Similarity Score

                # By default, the returned embeddings are np.array's
                embeddings = model.encode(responses, convert_to_tensor=True)

                # If `embeddings` is a torch.Tensor
                bert_sim = pairwise_cos_sim(embeddings, embeddings)

                # If `embeddings` is an np.array
                # bert_sim = cosine_similarity(embeddings)

                n_elements = bert_sim.shape[0] * (bert_sim.shape[0] - 1)
                avg_bert_sim = (
                                       bert_sim.sum() - bert_sim.diag().sum()).item() / n_elements
                results_df.loc[idx_row, "bert_sim"] = avg_bert_sim

                answer1_li = []
                answer2_li = []

                # Compute Jaccard Similarity Score
                unigram_jaccard = []
                bigram_jaccard = []

                try:
                    for i in range(len(responses_words)):
                        for j in range(i + 1, len(responses_words)):
                            unigram_jaccard += [
                                jaccard_sim(responses[i], responses[j], ngram=1)]
                            bigram_jaccard += [
                                jaccard_sim(responses[i], responses[j], ngram=2)]

                            answer1_li += [responses[i]]
                            answer2_li += [responses[j]]

                except:
                    traceback.print_exc()
                    continue

                # Calculate BERTScore
                t0 = time.time()

                P, R, F1 = score(answer1_li, answer2_li, transformer_model=transformer_model,
                                 tokenizer=tokenizer, device=DEVICE, model_type=model_type,
                                 rescale_with_baseline=True)

                P_d[row.name] = [P.cpu().numpy()]
                R_d[row.name] = [R.cpu().numpy()]
                F1_d[row.name] = [F1.cpu().numpy()]

                results_df.loc[idx_row, f"bertscore_P"] = torch.mean(P).item()
                results_df.loc[idx_row, f"bertscore_R"] = torch.mean(R).item()
                results_df.loc[idx_row, f"bertscore_F1"] = torch.mean(F1).item()
                print(f"\tBERTScore: {time.time() - t0:.2f} seconds")

                assert len(unigram_jaccard) == len(bigram_jaccard) == len(
                    responses) * (
                               len(responses) - 1) / 2

                avg_unigram_jaccard = np.mean(unigram_jaccard)
                avg_bigram_jaccard = np.mean(bigram_jaccard)

                results_df.loc[idx_row, "unigram_jaccard"] = avg_unigram_jaccard
                results_df.loc[idx_row, "bigram_jaccard"] = avg_bigram_jaccard

                # Step 6: Detect Contradictions - This is a complex task and usually requires manual review or advanced NLP techniques
                # For simplicity, let's just compare whether the responses are identical

                # Step 7: Calculate Answer Length Variability
                length = [len(resp) for resp in responses_words]

                # Calculate length variability
                length_mean = np.mean(length)
                length_std = np.std(length)
                results_df.loc[idx_row, "length_mean"] = length_mean
                results_df.loc[idx_row, "length_std"] = length_std

                # print(f"Average Semantic Similarity: {avg_semantic_similarity}")
                # print(f"Average Jaccard Similarity: {avg_jaccard_similarity}")
                # print(f"Average Cosine Similarity: {avg_cosine_similarity}")
                # print(f"Contradiction Rate: {contradiction_rate}")
                # print(f"Answer Length Variability: {length_variability}")
                # print(f"Unique Answer Rate: {unique_answer_rate}")

                if idx_row % 10 == 0:
                    save(results_df, path_results,
                         sheet_name=args.target_language)

            save(results_df, path_results, sheet_name=args.target_language)

    # with pd.ExcelWriter(path_results, engine='openpyxl', mode=mode) as writer:
    #
    #     results_df.to_excel(writer, index=False,
    #                         sheet_name=args.target_language)

    if args.do_topic_modeling:

        # ----------------- Topic Modeling -----------------
        # results_df_topic_modeling = pd.DataFrame()

        # results_df_topic_modeling[const.ID] = np.arange(0, len(df) * args.interval, args.interval)

        # results_df_topic_modeling.set_index(const.ID, inplace=True)

        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        df_topic_modeling = df.drop(const.ERROR, axis=1,
                                    errors="ignore").fillna("").dropna().copy()

        if df_topic_modeling.index.name != const.ID:
            df_topic_modeling.set_index(const.ID, inplace=True)

        documents = []

        idx_valid_rows = []

        for idx_row in range(0, len(df) * args.interval, args.interval):

            answers_to_one_question = []

            if idx_row not in df_topic_modeling.index:
                print(f"Skipping row {idx_row}")
                continue

            contains_only_stopwords = False
            for i in range(args.num_answers):
                if isinstance(df_topic_modeling.loc[idx_row, f"answer_{i}"],
                              str):
                    words = df_topic_modeling.loc[idx_row, f"answer_{i}"].split()

                    contains_only_stopwords = contains_only_stopwords or all(word.lower().strip() in stopwords_list for
                                                                             word in
                                                                             words)

                    answers_to_one_question.append(
                        df_topic_modeling.loc[idx_row, f"answer_{i}"])

            if not contains_only_stopwords:
                documents += answers_to_one_question
                idx_valid_rows += [idx_row]

        print(f"#Valid rows: {len(idx_valid_rows)}")
        assert len(documents) % args.num_answers == 0

        vectorizer = CountVectorizer()
        dtm = vectorizer.fit_transform(documents)
        idx_valid_rows = np.array(idx_valid_rows)

        # # ----------------- HDP -----------------

        if "hdp_mean" in results_df.columns and not (results_df.loc[:, "hdp_mean"].isna().all() or results_df.loc[:,
                                                                                       "hdp_std"].isna().all()):
            print(f"{args.dataset_name}\tSkipping HDP")

        else:
            # Create a dictionary representation of the documents
            from gensim.models import HdpModel
            from gensim.corpora import Dictionary
            from gensim.utils import simple_preprocess

            # Preprocess the documents
            texts = [simple_preprocess(doc) for doc in documents]
            dictionary = Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]

            # Fit the HDP model
            hdp = HdpModel(corpus, dictionary, random_state=args.seed)

            # Get the topic distribution for each document
            topic_dist_d = [dict(hdp[doc]) for doc in corpus]

            # Convert to matrix for similarity calculation
            topic_dist = np.zeros((len(corpus), hdp.get_topics().shape[0]))
            for doc_idx, doc_topic_dist in enumerate(topic_dist_d):
                for topic_idx, prob in doc_topic_dist.items():
                    topic_dist[doc_idx][topic_idx] = prob

            topic_dist = topic_dist.reshape(len(idx_valid_rows),
                                            args.num_answers,
                                            hdp.get_topics().shape[0])

            topic_dist = torch.tensor(topic_dist, device=args.device).unsqueeze(
                2)

            # Calculate cosine similarity between pairs of documents

            cos_sim = F.cosine_similarity(topic_dist,
                                          topic_dist.transpose(1, 2), dim=3)

            mean, std = get_topic_mean_std(cos_sim)

            results_df.loc[idx_valid_rows, f"hdp_mean"] = mean
            results_df.loc[idx_valid_rows, f"hdp_std"] = std

        # ----------------- Top2Vec -----------------

        """
        # This does not work for now
        
        if f"top2vec_mean" in results_df.columns and f"top2vec_std" in results_df.columns:
            print(f"{args.dataset_name}\tSkipping Top2Vec")

        else:

            model = Top2Vec(documents, speed="learn", workers=args.num_workers, verbose=True)

            # Get document vectors
            document_vectors = model._get_document_vectors()

            # Compute cosine similarity
            cosine_sim = cosine_similarity(document_vectors)

            # Initialize a matrix with zeros for storing the topic probabilities
            topic_distribution = np.zeros(
                (len(df_topic_modeling), model.get_num_topics()))

            # Calculate the probabilities
            for topic in range(model.get_num_topics()):
                _, document_scores, document_ids = model.search_documents_by_topic(
                    topic_num=topic, num_docs=len(df_topic_modeling))
                for doc_score, doc_id in zip(document_scores, document_ids):
                    topic_distribution[doc_id, topic] = doc_score
                    
        """

        # ----------------- LDA -----------------

        for num_topics in [10, 20, 50, 100, 200, 500]:

            if not (results_df.loc[:, f"lda{num_topics}_mean"].isna().all() or results_df.loc[:, f"lda{num_topics}_std"].isna().all()):
                print(f"{args.dataset_name}\tSkipping #Topic {num_topics}")
                continue

            lda = LatentDirichletAllocation(n_components=num_topics,
                                            random_state=args.seed,
                                            n_jobs=args.num_workers)
            print(f"Fitting LDA models with {num_topics} topics")
            lda.fit(dtm)

            topic_dist = lda.transform(dtm)

            topic_dist = topic_dist.reshape(len(idx_valid_rows),
                                            args.num_answers, num_topics)

            topic_dist = torch.tensor(topic_dist, device=args.device).unsqueeze(
                2)
            topic_dist_t = topic_dist.transpose(1, 2)

            cos_sim = F.cosine_similarity(topic_dist, topic_dist_t, dim=3)

            mean, std = get_topic_mean_std(cos_sim)

            results_df.loc[idx_valid_rows, f"lda{num_topics}_mean"] = mean
            results_df.loc[idx_valid_rows, f"lda{num_topics}_std"] = std

            # for col in results_df_topic_modeling.columns:
            #     if col.startswith("topics"):
            #         results_df[col] = results_df_topic_modeling[col]

        save(results_df, path_results,
             sheet_name=args.target_language)

        # -------------- [END] Topic Modeling -----------------


if __name__ == "__main__":

    # TODO
    args.do_similarity = False
    args.do_topic_modeling = True
    project_setup(args)
    openai_setup(args)

    # for language in ["Chinese"]:

    TEMPERATURES = [0.0, 0.25, 0.5, 0.75, 1.0]
    TEMPERATURES = [0.0, 1.0]

    for temperature in TEMPERATURES:
        for language in ["English", "Spanish", "Chinese", "Hindi", ]:

            args.temperature = temperature
            args.target_language = language

            consistency(language=args.target_language)
