from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer


class WordTypeMixin:
    def __init__(self):
        light_verb = {'do', 'give', 'have', 'make', 'get', 'take', 'use', 'go',
                      'does', 'doing', 'did', 'done',
                      'gives', 'giving', 'gave', 'given',
                      'has', 'having', 'had',
                      'makes', 'making', 'made',
                      'gets', 'getting', 'got', 'gotten',
                      'takes', 'taking', 'took', 'taken',
                      'uses', 'using', 'used',
                      'goes', 'going', 'went', 'gone'}
        other_verb = {'appear', 'appears', 'appeared', 'appearing', 'am', 'be', 'is', 'are', 'was', 'been', 'were',
                      'being', 'seem', 'seems', 'seemed', 'seeming', 'let', 'lets', 'letting', 'bring', 'brings',
                      'brought', 'bringing', 'put', 'puts', 'putting', 'show', 'shown', 'shows', 'showed',
                      'showing', 'came', 'come', 'comes', 'coming'}
        self.all_light_verb = light_verb.union(other_verb)
        super().__init__()

    @staticmethod
    def is_root_nsubj(tok, root):
        while tok.idx != root.idx and tok.head_idx != tok.idx:
            if tok.dep_rel in ['nsubj', 'auxpass', 'aux', 'nsubjpass']:
                tok = tok.head
            else:
                break
        return tok.idx == root.idx

    @staticmethod
    def is_possibly_nominal(tok):
        if tok.dep_rel in {'nsubj', 'iobj', 'dobj',
                           'nsubjpass'}:  # 'attr', 'intj']: don't appear in Universal dependency
            return True
        if (tok.dep_rel == 'pobj' or tok.dep_rel.startswith("nmod")) and \
                not (tok.tag.startswith("VB") and tok.head.tag in {"POS", "TO"}):
            return True
        if tok.dep_rel in {'dative', 'agent', 'pcomp', 'ccomp', 'xcomp', 'acomp', 'npadvmod', 'oprd', "ROOT"} \
                and (tok.tag in {"NN", "NNP", "NNS", "NNPS", 'EX', "PRP", "WP", 'CD'}):
            return True
        if tok.dep_rel in {'conj', 'appos'} and tok.head_idx != tok.idx and WordTypeMixin.is_possibly_nominal(tok.head) \
                and (tok.tag == tok.head.tag or tok.tag in {"NN", "NNP", "NNS", "NNPS", 'EX', "PRP", "WP",
                                                            'CD'}):
            return True
        return False

    @staticmethod
    def is_possibly_predicate(tok):
        if tok.dep_rel in {'csubj', 'advcl', 'acl', 'relcl', 'ROOT'} and tok.tag in {'MD', 'VB', 'VBD', 'VBG', 'VBN',
                                                                                     'VBP',
                                                                                     'VBZ', 'JJ', 'JJR', 'JJS', 'RB',
                                                                                     'RBR',
                                                                                     'RBS', 'WRB', 'NN', 'NNS'}:
            return True
        if tok.dep_rel in {'acomp', 'ccomp', 'xcomp', 'oprd', 'pcomp'} and tok.tag in {'MD', 'VB', 'VBD', 'VBG', 'VBN',
                                                                                       'VBP', 'VBZ', 'JJ', 'JJR', 'JJS',
                                                                                       'RB', 'RBR', 'RBS', 'WRB'}:
            return True
        if tok.dep_rel == 'pobj' and tok.tag in {'MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'} and tok.head.tag in {
            "POS",
            "TO"}:
            return True
        if tok.dep_rel in {'conj'} and (tok.tag == tok.head.tag or tok.tag in {'MD', 'VB', 'VBD', 'VBG', 'VBN',
                                                                               'VBP', 'VBZ', 'JJ', 'JJR', 'JJS',
                                                                               'RB', 'RBR', 'RBS', 'WRB'}) \
                and tok.head_idx != tok.idx and WordTypeMixin.is_possibly_predicate(tok.head):
            return True
        return False

    @staticmethod
    def is_possibly_prep(tok):
        if tok.dep_rel in {'case'} and tok.tag in {'IN', 'RP', "POS", "TO", 'RB', 'RBR', 'RBS', 'WRB', 'MD', 'VB',
                                                   'VBD',
                                                   'VBG', 'VBN', 'VBP', 'VBZ'}:
            return True
        if tok.dep_rel in {'dative', 'agent', 'ccomp', 'xcomp', 'advmod', 'pcomp', 'prt'} and tok.tag in {'IN', 'RP'}:
            return True
        if tok.dep_rel in {'advmod', 'prt'} and tok.tag in {'RB', 'RBR', 'RBS', 'WRB'}:
            return True
        if tok.dep_rel in {'conj'} and tok.head_idx != tok.idx and tok.tag in {'IN', 'RP'} \
                and WordTypeMixin.is_possibly_prep(tok.head):
            return True
        return False

    @staticmethod
    def is_modifier(token):
        modifier_deps = {'advmod', 'neg', 'mark', 'punct', 'prt', 'predet', 'nmod', 'cc', 'npadvmod', 'dep',
                         'nummod', 'amod', 'case', 'det', 'compound', 'poss', 'expl', 'quantmod', 'aux', 'auxpass'}
        subtree = []
        WordTypeMixin.get_subtree(token, subtree)

        sub_deps = [t.dep_rel in modifier_deps or (t.dep_rel == 'conj' and t.head.dep_rel in modifier_deps) or
                    (token.dep_rel == 'appos' and token.text == 'all') for t in subtree]
        return all([c for c in sub_deps])

    @staticmethod
    def get_subtree(token, subtree):
        for t in token.children:
            WordTypeMixin.get_subtree(t, subtree)
            subtree.append(t)

    def is_possibly_light(self, tok):
        return tok.tag in {'MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'} and \
               tok.text in self.all_light_verb


class WordNetMixin:
    def __init__(self):
        self.wn_cache = {}
        super().__init__()

    def wn_query(self, x, pos=None):
        key = (x, pos)
        if key not in self.wn_cache:
            self.wn_cache[key] = wn.synsets(x, pos)
        return [c for c in self.wn_cache[key]]


class LemmatizerMixin:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        super().__init__()

    @staticmethod
    def correct_mislemmas(text, lemma):
        mislemmas = {'taxes': 'tax', 'lenses': 'lens', 'goods': 'goods', 'waters': 'waters', 'ashes': 'ash',
                     'fries': 'fries', 'politics': 'politics', 'glasses': 'glasses', 'clothes': 'clothes',
                     'scissors': 'scissors', 'shorts': 'shorts', 'thanks': 'thanks',
                     'media': 'media', 'woods': 'woods', 'data': 'data', 'belongings': 'belongings'}
        if text not in mislemmas:
            return lemma
        return mislemmas[text]

    def lemmatize(self, token, pos=None):
        if pos:
            lemma = self.lemmatizer.lemmatize(token.text, pos)
        else:
            lemma = self.lemmatizer.lemmatize(token.text)
        lemma = self.correct_mislemmas(token.text, lemma)
        return lemma
