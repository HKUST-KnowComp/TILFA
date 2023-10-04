import os
import json
import argparse
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from aser_tool.concept_result import ConceptResult


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str,
                        default="./gun_control_train_pure_wordnet.json")
    parser.add_argument("--output_dir", type=str,
                        default="./gloss_input")
    args = parser.parse_args()
    output_file_name = args.input_file_path.split("/")[-1].rsplit(".", maxsplit=1)[0] + ".csv"
    args.output_file_path = os.path.join(args.output_dir, output_file_name)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.input_file_path) as fin:
        variation_list = [json.loads(line) for line in fin]

    fw = open(args.output_file_path, 'w')
    fw.write('target_id	label	sentence	gloss	sense_key\n')

    for i, var in tqdm(enumerate(variation_list), "building gloss bert input"):
        text_id = var["id"]
        cr = ConceptResult.load_from_dict(var)
        te, word, l_i = cr.token_list, cr.instance, cr.start_idx
        r_i, syn_list, src = cr.end_idx, cr.synset_list, cr.source
        if src != "wn":
            continue
        sent = te[: l_i] + ['"'] + te[l_i: r_i + 1] + ['"'] + te[r_i + 1:] + ['.']
        sent = " ".join(sent)
        sent = sent.replace('PersonX', 'Alice').replace('PersonY', 'Bob').replace('PersonZ', 'Charlie')
        for j, synset in enumerate(syn_list):
            id = '%d|%d' % (text_id, j)
            gloss = word + ' : ' + wn.synset(synset).definition()
            fw.write('\t'.join([id, '0', sent, gloss, synset]) + '\n')
    fw.close()
