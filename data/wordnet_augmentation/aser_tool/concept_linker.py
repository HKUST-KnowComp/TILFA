import json
from tqdm import tqdm
from .POSITIVE_RULES import ALL_POS_EVENTUALITY_RULES
from .POSITIVE_RULES import NOMINAL_SAVED_COMPONENTS, VERBAL_SAVED_COMPONENTS
from collections import OrderedDict
from .match_util import WordNetMixin, WordTypeMixin, LemmatizerMixin
from .concept_result import ConceptResult
from .probase_code.probase_client import ProbaseClient


class ConceptLinker(WordTypeMixin, WordNetMixin, LemmatizerMixin):
    def __init__(self, G_aser):
        self.G_aser = G_aser
        self.pb = ProbaseClient()
        self.indefinite_pronoun = {'another', 'anybody', 'anyone', 'anything',
                                   'each', 'either', 'enough', 'everybody',
                                   'everyone', 'everything', 'less', 'little',
                                   'much', 'neither', 'nobody', 'no one', 'nothing',
                                   'one', 'other', 'somebody', 'someone', 'something',
                                   'both', 'few', 'fewer', 'many', 'others', 'several'}
        self.transparent_noun_prep = json.load(open("/home/data/zwanggy/data_atomic/nomlex.transparent.json"))
        super().__init__()

    @staticmethod
    def deduplicate_mentions(all_linked_concept):
        all_deduplicate_concept = []
        for men_list in all_linked_concept:
            cur_concept_list = []
            for i, var_i in enumerate(men_list):
                # adjacent event should be not the same
                covered_flag = False
                for j, var_j in enumerate(men_list):
                    if i == j or var_i["head_idx"] != var_j["head_idx"]:
                        continue
                    if var_i["start_idx"] >= var_j["start_idx"] and var_i["end_idx"] <= var_j["end_idx"]:
                        covered_flag = True
                        break
                if not covered_flag:
                    cur_concept_list.append(var_i)
            all_deduplicate_concept.append(cur_concept_list)
        return all_deduplicate_concept

    @staticmethod
    def match_head_words_by_dependency(root, pattern, head_word_type):
        positive_rule_list = ALL_POS_EVENTUALITY_RULES[pattern].positive_rules
        # extend phrases by matching positive rules
        asertag2token = OrderedDict([(positive_rule_list[0][0], root)])  # head, relation, tail

        for pos_rule in positive_rule_list:
            head_tag, relation, tail_tag = pos_rule
            head_word = asertag2token[head_tag]
            for children in head_word.children:
                if children.dep_rel == relation:
                    assert tail_tag not in asertag2token
                    asertag2token[tail_tag] = children
        if head_word_type == "nominal":
            saved_set = NOMINAL_SAVED_COMPONENTS[pattern]
        elif head_word_type == "verbal":
            saved_set = VERBAL_SAVED_COMPONENTS[pattern]
        elif head_word_type == "event":
            saved_set = {positive_rule_list[0][0]}
        else:
            raise ValueError("wrong head word type")
        asertag2token = {key: value for key, value in asertag2token.items() if key in saved_set}
        return [value.idx for key, value in asertag2token.items()]

    @staticmethod
    def deduplicate_by_keys(var_list, key_func):
        key2var_dict = {}
        priority = {"none": 1, "verb phrase": 2, "pb": 3, "wn": 4}
        for var in var_list:
            key = key_func(var)
            if key not in key2var_dict:
                key2var_dict[key] = var
            elif priority[var["source"]] > priority[key2var_dict[key]["source"]]:
                key2var_dict[key] = var
        # get back to list
        new_var_list = []
        for key, var in key2var_dict.items():
            new_var_list.append(var)
        new_var_list = sorted(new_var_list, key=lambda x: (x["start_idx"], x["end_idx"]))
        return new_var_list

    def match_components(self, parsed_event_list, pattern, head_word_type):
        node_components_list = []
        for parsed_event in tqdm(parsed_event_list, "finding components"):
            root = [t for t in parsed_event.token_list if t.dep_rel == 'ROOT'][0]
            if pattern is not None:
                cur_pattern = pattern
            else:
                cur_pattern = self.G_aser.nodes[parsed_event.event_text]["info"]["pattern"]
            root_head_list = self.match_head_words_by_dependency(root, cur_pattern, head_word_type)
            node_components_list.append(root_head_list)

        return node_components_list

    def match_concepts(self, parsed_event_list, components, head_word_type):
        if head_word_type in {"nominal", "verbal"}:
            all_linked_concepts = []
            for i, parsed_node in enumerate(tqdm(parsed_event_list)):
                linked_concepts = []
                for c_i in components[i]:
                    if head_word_type == "nominal":
                        variations = self.collect_nominal_candidate(parsed_node, c_i)
                    elif head_word_type == "verbal":
                        variations = self.collect_verbal_candidate(parsed_node, c_i)
                    else:
                        raise ValueError("Wrong head word type")
                    linked_concepts.extend(variations)
                linked_concepts = self.deduplicate_by_keys(linked_concepts,
                                                           key_func=lambda x: (x["start_idx"], x["end_idx"]))
                all_linked_concepts.append(linked_concepts)
            return all_linked_concepts
        elif head_word_type == "event":
            all_linked_concepts = []
            for i, parsed_node in enumerate(tqdm(parsed_event_list)):
                var = self.collect_event_candidate(parsed_node)
                all_linked_concepts.append(var)
            return all_linked_concepts

    def merge_nominal_pred_match(self, nom_sub, pred_sub):
        nom_texts = set(n.instance for n in nom_sub)
        results = [x for x in nom_sub]  # shallow copy
        for p in pred_sub:
            if p.instance not in nom_texts:
                results.append(p)
        return results

    def collect_nominal_candidate(self, parsed_event, c_i):
        tuple_results = []
        head_text = parsed_event.token_list[c_i].text
        if head_text in self.indefinite_pronoun:
            return tuple_results

        nominal_sub = self.match_nominal_candidate(parsed_event, c_i) if parsed_event.token_list[c_i].is_nominal else []
        results = list(reversed(nominal_sub))
        for i, res in enumerate(results):
            synset_list = []
            if res.source == "wn":
                for s in res.synset_list:
                    synset_list.append(s.name())
                synset_list = sorted(synset_list)
            elif res.source == "pb":
                synset_list = [word for word, info in res.synset_list]
            elif res.source in {"none", "verb phrase"}:
                pass
            else:
                raise KeyError("Wrong source")
            res.synset_list = synset_list
            tuple_results.append(res.save_to_dict())
        return tuple_results

    def collect_verbal_candidate(self, parsed_event, c_i):
        tuple_results = []
        head_text = parsed_event.token_list[c_i].text
        if head_text in self.indefinite_pronoun:
            return tuple_results

        verbal_sub = self.match_verbal_candidate(parsed_event, c_i) if parsed_event.token_list[c_i].is_predicate else []
        results = list(reversed(verbal_sub))
        for i, res in enumerate(results):
            synset_list = []
            if res.source == "wn":
                for s in res.synset_list:
                    synset_list.append(s.name())
                synset_list = sorted(synset_list)
            elif res.source == "pb":
                synset_list = [word for word, info in res.synset_list]
            elif res.source in {"none", "verb phrase"}:
                pass
            else:
                raise KeyError("Wrong source")
            res.synset_list = synset_list
            tuple_results.append(res.save_to_dict_with_synset_source())
        return tuple_results

    def match_verbal_candidate(self, parsed_event, c_i):
        token_list = parsed_event.token_list
        token = token_list[c_i]
        results = []
        ch_list = token.children

        # light verb: If we find a light verb, we just return empty
        if self.is_possibly_light(token):
            return results

        # In these special constructions we ignore the multi-word case and leave it to annotators
        # the following "for" statement is written for the passive voice, for example, "is able to finish my homework"
        if token.text in {'certain', 'meant', 'able', 'obliged', 'forced',
                          'set', 'allowed', 'going', 'supposed', 'bound', 'likely', 'sure'}:
            for ch in ch_list:
                if ch.dep_rel in ['ccomp', 'xcomp'] and ch.is_predicate:
                    if all(t.text != "to" or t.tag != "TO" for t in ch.children):
                        continue
                    results.extend(self.match_verbal_candidate(parsed_event, ch.idx))

                    for gch in ch.children:
                        if gch.dep_rel == "conj" and gch.is_predicate:
                            results.extend(self.match_verbal_candidate(parsed_event, gch.idx))
                    return results

        lemma = self.lemmatize(token)
        syn = self.wn_query(token.text, 'v')
        if not syn:
            syn = self.wn_query(lemma, 'v')
        if token.tag not in {'MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'} or not syn or token.text.endswith('ing'):
            syn_full = self.wn_query(token.text)
            if not syn_full:
                syn_full = self.wn_query(lemma)
            syn = list(set(syn).union(syn_full))
        if syn:
            results.append(ConceptResult(syn, token.text, c_i, c_i, c_i, "wn", parsed_event.event_text,
                                         parsed_event.text_token_list, parsed_event.pattern))
        return results

    def collect_event_candidate(self, parsed_event):
        c_i = None
        for idx, token in enumerate(parsed_event.token_list):
            if token.dep_rel == "ROOT":
                c_i = idx
                break
        cr = ConceptResult([], parsed_event.event_text, 0, len(parsed_event.token_list),
                           c_i, "root phrase", parsed_event.event_text,
                           parsed_event.text_token_list, parsed_event.pattern)
        return [cr.save_to_dict()]

    @staticmethod
    def get_full_phrase(token_list, marks, no_det_prep=True):
        token_list = [token for token, m in zip(token_list, marks) if m]
        # get full phrase without DET and prepositions, like a, an, the, of.
        if no_det_prep and (token_list[0].dep_rel in {'det', 'nmod:poss', 'case'} or
                            token_list[0].tag in {'DT', 'PDT', 'PRP$', 'WDT', 'WP$', 'EX',
                                                  'IN'}):  # delete DET and Prep
            token_list.pop(0)
        if len(token_list) > 1 and token_list[0].text in {"'s"}:
            token_list.pop(0)
        token_list = sorted(token_list, key=lambda t: t.idx)
        l, r = token_list[0].idx, token_list[-1].idx
        token_list = [tok.text for tok in token_list]
        return " ".join(token_list), l, r

    @staticmethod
    def get_left_edge(token):
        while True:
            c = token.children
            c_idx = [i.idx for i in c]
            c = [i for _, i in sorted(zip(c_idx, c))]
            if len(c) == 0 or c[0].idx > token.idx:
                return token
            else:
                token = c[0]

    @staticmethod
    def get_right_edge(token, exclude_conj=False):
        c = token.children
        c_idx = [i.idx for i in c]
        c = [i for _, i in sorted(zip(c_idx, c))]
        if exclude_conj:
            for k in range(len(c)):
                if c[k].dep_rel in ['cc', 'conj']:
                    c = c[:k]
                    break
        while True:
            if len(c) == 0 or c[-1].idx < token.idx:
                return token
            else:
                token = c[-1]
            c = token.children

    def collect_modifier_marks(self, parsed_event, tok, c_i):
        le = self.get_left_edge(tok).idx
        re = self.get_right_edge(tok).idx
        mark = [False for _ in parsed_event.token_list]
        mark[c_i] = True
        for i in range(le, re + 1):
            if i == c_i or not WordTypeMixin.is_modifier(parsed_event.token_list[i]):
                continue
            tok_i = i
            # check the root of token i
            while WordTypeMixin.is_modifier(parsed_event.token_list[tok_i]) and tok_i != c_i:
                tok_i = parsed_event.token_list[tok_i].head_idx
            if tok_i == c_i:
                mark[i] = True
            if parsed_event.pattern in {"s-be-o", "s-v-be-o", "s-v-o-be-o"} and \
                    parsed_event.token_list[i].head_idx == c_i and parsed_event.token_list[i].dep_rel in {"nsubj",
                                                                                                          "cop"}:
                mark[i] = False
        return mark

    def match_by_substr(self, parsed_event, tok, c_i, text, marks, is_transparent):
        results = []
        token_list = parsed_event.token_list
        le = self.get_left_edge(tok).idx
        re = self.get_right_edge(tok).idx
        for l in range(le, c_i + 1):
            if marks is not None and not marks[l]:
                continue
            if token_list[l].dep_rel in ['det', 'poss']:
                continue
            for r in range(c_i, re + 1):
                if marks is not None and not marks[r]:
                    continue
                if c_i - l > 1 or r - c_i > 1:
                    continue
                cand_subs = []
                lemma = self.lemmatize(token_list[c_i])
                for mid_text in [lemma, token_list[c_i].text]:
                    sub_text = text[token_list[l].offset: token_list[c_i].offset] + mid_text + \
                               text[token_list[c_i].offset + len(token_list[c_i].text): token_list[r].offset + len(
                                   token_list[r].text)]
                    cand_subs.append(sub_text)

                syn = self.wn_query(cand_subs[1], 'n')
                if not syn:
                    syn = self.wn_query(cand_subs[0], 'n')
                if tok.tag not in {"NN", "NNP", "NNS", "NNPS"} or not syn:
                    syn_full = self.wn_query(cand_subs[1])
                    if not syn_full:
                        syn_full = self.wn_query(cand_subs[0])
                    syn = list(set(syn).union(syn_full))
                if syn:
                    results.append(ConceptResult(syn, cand_subs[1], l, r, c_i, "wn", parsed_event.event_text,
                                                 parsed_event.text_token_list, parsed_event.pattern))
                else:
                    pb_resp = self.pb.query(cand_subs[1], 'mi', 10)
                    if not pb_resp:
                        pb_resp = self.pb.query(cand_subs[0], 'mi', 10)
                    if pb_resp:
                        results.append(ConceptResult(pb_resp, cand_subs[1], l, r, c_i, "pb", parsed_event.event_text,
                                                     parsed_event.text_token_list, parsed_event.pattern))

        # check the full phrase
        # head + all modifier
        full_phrase, l, r = self.get_full_phrase(token_list, marks)
        cur_phrase_set = {res.instance for res in results}
        if full_phrase not in cur_phrase_set:
            results.append(ConceptResult([], full_phrase, l, r, c_i, "none", parsed_event.event_text,
                                         parsed_event.text_token_list, parsed_event.pattern))

        # for transparent constructions, like a lot of xxx, delete the linking of 'lot'
        # Also, for "an amount of", delete "amount" and "amount of"
        # So, currently, we only save full_phrase
        if is_transparent:
            results = [cr for cr in results if cr.instance == full_phrase]

        return results

    def match_nominal_candidate(self, parsed_event, index):
        token_list = parsed_event.token_list
        token = token_list[index]
        if token.text in {'PersonX', 'PersonY', 'PersonZ'}:
            return []

        node_text = parsed_event.event_text
        results = []
        children = token.children

        # a lot (head) of money
        transparent_preps = self.transparent_noun_prep.get(token.text, [])
        if token.tag in {'DT', 'PDT', 'PRP$', 'WDT', 'WP$', 'EX', 'PRP', 'WP', 'CD'}:  # DET, PRONOUN, and count
            transparent_preps.append('of')
        is_transparent = "of" in transparent_preps

        # detect transparent constructions
        right_component_list = []
        if index + 1 < len(token_list) and token.text == "lot" and token_list[index + 1].text == "of":
            right_component_list.append(index + 2)
        else:
            for ch in children:
                if ch.is_nominal and ch.dep_rel == "nmod:of":
                    right_component_list.append(ch.idx)
                    is_transparent = True
                    break
        if right_component_list:
            ch = token_list[right_component_list[0]]
            for gch in ch.children:
                if gch.dep_rel.startswith("conj") and gch.is_nominal:
                    right_component_list.append(gch.idx)

        for right_idx in right_component_list:
            results.extend(self.match_nominal_candidate(parsed_event, right_idx))
        # s-v-o s-be-o
        if token.tag not in {'DT', 'PDT', 'PRP$', 'WDT', 'WP$', 'EX', 'PRP', 'WP'}:  # all DET and PRONOUN
            marks = self.collect_modifier_marks(parsed_event, token, index)
            # get all modifiers in the subtree of the current noun
            collected = self.match_by_substr(parsed_event, token, index, node_text, marks, is_transparent)
            results.extend(collected)

        return results
