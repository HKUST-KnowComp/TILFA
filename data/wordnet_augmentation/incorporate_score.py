import json
import argparse
import os.path

from tqdm import tqdm
from nltk.corpus import wordnet as wn
import spacy


def get_name(synset):
    name = synset.get_name()
    name = name.split(".")[0]
    return name


def get_all_words(synset, query, query_lemma):
    word_set = set(lemma.name().replace("_", " ") for lemma in synset.lemmas())
    word_set.discard(query)
    word_set.discard(query_lemma)
    return word_set


def merge_words(list_1, set_2, max_size):
    count = 0
    for item in set_2:
        if len(list_1) > max_size:
            break
        name = item.split(".")[0]
        if name not in list_1:
            list_1.append(name)
            count += 1
    return list_1, count


def lemmatize_with_spacy(cur_var, spacy_nlp):
    if cur_var["start_idx"] != cur_var["end_idx"]:
        return cur_var["instance"]
    event_text = cur_var["event_text"]
    doc = spacy_nlp(event_text)

    offset = 0
    for i in range(cur_var["head_idx"]):
        offset += len(cur_var["token_list"][i]) + 1
    for token in doc:
        if token.idx == offset:
            break
    return token.lemma_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", help="linking and conceptualization file without scores",
                        default="./gun_control_train_pure_wordnet.json")
    parser.add_argument("--score_file_path", help="WSD scores",
                        default="./gloss_score/gun_control_train_pure_wordnet/results.txt")
    parser.add_argument("--output_file_dir", help="concepts filtered by WSD scores",
                        default="./final/")
    # parser.add_argument("--csv_dir", help="csv output format",
    #                     default="/home/data/zwanggy/matching_concept/verbal_random_nyt/final_csv/")
    parser.add_argument("--top_k_sense", help="get hypernyms from top k senses",
                        default=1)
    parser.add_argument("--max_cand", help="max number of candidates",
                        default=10)
    args = parser.parse_args()
    input_file_name = args.input_file_path.split("/")[-1]
    csv_file_name = input_file_name.rsplit(".", maxsplit=1)[0] + ".csv"
    args.output_file_path = os.path.join(args.output_file_dir, input_file_name)
    # args.csv_path = os.path.join(args.csv_dir, csv_file_name)
    if not os.path.exists(args.output_file_dir):
        os.makedirs(args.output_file_dir)
    # if not os.path.exists(args.csv_dir):
    #     os.makedirs(args.csv_dir)

    nlp = spacy.load('en_core_web_sm')

    with open(args.score_file_path) as fin:
        score_list = [float(line.split()[-1]) for line in fin]

    with open(args.input_file_path) as fin:
        variation_list = [json.loads(line) for line in fin]

    scored_variation = []

    start_idx = 0
    for var in tqdm(variation_list):
        syn_list = var["synset_list"]
        if var["source"] != "wn":
            var["hypernym_list"] = []
            var["entailment_list"] = []
            var["definition_list"] = []
            scored_variation.append(var)
            continue
        synset_number = len(syn_list)
        cur_score_list = score_list[start_idx: start_idx + synset_number]
        start_idx = start_idx + synset_number
        syn_list = [synset for _, synset in sorted(zip(cur_score_list, syn_list), reverse=True)]
        word_lemma = lemmatize_with_spacy(var, nlp)

        # # get entailment
        # entailment_list = []
        # entailment_synset_list = [entailment_synset for synset in syn_list[: args.top_k_sense]
        #                            for entailment_synset in wn.synset(synset).entailments()]
        # for entailment_synset in entailment_synset_list:
        #     cur_syn_words = get_all_words(entailment_synset, var["instance"], word_lemma)
        #     entailment_list, _ = merge_words(entailment_list, cur_syn_words, args.max_cand)
        # if len(entailment_list) >= args.max_cand:
        #     entailment_list = entailment_list[: args.max_cand]

        synonym_list = []
        for cur_synset in syn_list[: args.top_k_sense]:
            cur_synset = wn.synset(cur_synset)
            cur_syn_words = get_all_words(cur_synset, var["instance"], word_lemma)
            synonym_list, _ = merge_words(synonym_list, cur_syn_words, args.max_cand)
            if len(synonym_list) >= args.max_cand:
                synonym_list = synonym_list[: args.max_cand]

        var["synonym_list"] = synonym_list

        scored_variation.append(var)


    assert start_idx == len(score_list)

    # the csv header:
    # event, marked words, left index, right index, source,

    with open(args.output_file_path, "w") as fout:
        for var in scored_variation:
            fout.write(json.dumps(var) + "\n")

    # # the csv header:
    # # event, marked words, left index, right index, source,
    # df = pd.DataFrame(scored_variation)
    # df.to_csv(args.csv_path, index=False)
