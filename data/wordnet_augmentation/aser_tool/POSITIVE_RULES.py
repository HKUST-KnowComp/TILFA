from collections import defaultdict


class EventualityRule(object):
    def __init__(self):
        self.positive_rules = list()
        self.possible_rules = list()
        self.negative_rules = list()


ALL_POS_EVENTUALITY_RULES = dict()

NOMINAL_SAVED_COMPONENTS = defaultdict(set)
VERBAL_SAVED_COMPONENTS = defaultdict(set)

# nsubj-verb
E_rule = EventualityRule()
E_rule.positive_rules.append(("V1", "nsubj", "S1"))
ALL_POS_EVENTUALITY_RULES["s-v"] = E_rule
NOMINAL_SAVED_COMPONENTS["s-v"] = {"S1"}
VERBAL_SAVED_COMPONENTS["s-v"] = {"V1"}

# subj-verb-dobj
E_rule = EventualityRule()
E_rule.positive_rules.append(("V1", "nsubj", "S1"))
E_rule.positive_rules.append(("V1", "dobj", "O1"))
ALL_POS_EVENTUALITY_RULES["s-v-o"] = E_rule
NOMINAL_SAVED_COMPONENTS["s-v-o"] = {"S1", "O1"}
VERBAL_SAVED_COMPONENTS["s-v-o"] = {"V1"}

# subj-verb-adjective
E_rule = EventualityRule()
E_rule.positive_rules.append(("V1", "nsubj", "S1"))
E_rule.positive_rules.append(("V1", "xcomp", "A1"))
ALL_POS_EVENTUALITY_RULES["s-v-a"] = E_rule
NOMINAL_SAVED_COMPONENTS["s-v-a"] = {"S1"}
VERBAL_SAVED_COMPONENTS["s-v-a"] = {"V1"}

# subj-verb-verb
E_rule = EventualityRule()
E_rule.positive_rules.append(("V1", "nsubj", "S1"))
E_rule.positive_rules.append(("V1", "xcomp", "V2"))
ALL_POS_EVENTUALITY_RULES["s-v-v"] = E_rule
NOMINAL_SAVED_COMPONENTS["s-v-v"] = {"S1"}
VERBAL_SAVED_COMPONENTS["s-v-v"] = {"V1", "V2"}

# subj-verb-dobj1-dobj2
E_rule = EventualityRule()
E_rule.positive_rules.append(("V1", "nsubj", "S1"))
E_rule.positive_rules.append(("V1", "dobj", "O1"))
E_rule.positive_rules.append(("V1", "iobj", "O2"))
ALL_POS_EVENTUALITY_RULES["s-v-o-o"] = E_rule
NOMINAL_SAVED_COMPONENTS["s-v-o-o"] = {"S1", "O1", "O2"}
VERBAL_SAVED_COMPONENTS["s-v-o-o"] = {"V1"}

# subj-verb-verb-object
E_rule = EventualityRule()
E_rule.positive_rules.append(("V1", "nsubj", "S1"))
E_rule.positive_rules.append(("V1", "xcomp", "V2"))
E_rule.positive_rules.append(("V2", "dobj", "O1"))
ALL_POS_EVENTUALITY_RULES["s-v-v-o"] = E_rule
NOMINAL_SAVED_COMPONENTS["s-v-v-o"] = {"S1", "O1"}
VERBAL_SAVED_COMPONENTS["s-v-v-o"] = {"V1", "V2"}

# subj-verb-dobj1-verb-dobj2
E_rule = EventualityRule()
E_rule.positive_rules.append(("V1", "nsubj", "S1"))
E_rule.positive_rules.append(("V1", "dobj", "O1"))
E_rule.positive_rules.append(("V1", "xcomp", "V2"))
E_rule.positive_rules.append(("V2", "dobj", "O2"))
ALL_POS_EVENTUALITY_RULES["s-v-o-v-o"] = E_rule
NOMINAL_SAVED_COMPONENTS["s-v-o-v-o"] = {"S1", "O1", "O2"}
VERBAL_SAVED_COMPONENTS["s-v-o-v-o"] = {"V1", "V2"}

# nsubj-verb1-dobj1-verb2-dobj2-iobj3
E_rule = EventualityRule()
E_rule.positive_rules.append(("V1", "nsubj", "S1"))
E_rule.positive_rules.append(("V1", "dobj", "O1"))
E_rule.positive_rules.append(("V1", "xcomp", "V2"))
E_rule.positive_rules.append(("V2", "dobj", "O2"))
E_rule.positive_rules.append(("V2", "iobj", "O3"))
ALL_POS_EVENTUALITY_RULES["s-v-o-v-o-o"] = E_rule
NOMINAL_SAVED_COMPONENTS['s-v-o-v-o-o'] = {"S1", "O1", "O2", "O3"}
VERBAL_SAVED_COMPONENTS["s-v-o-v-o-o"] = {"V1", "V2"}

# subj-be-adjective
E_rule = EventualityRule()
E_rule.positive_rules.append(("A1", "^cop", "V1"))
E_rule.positive_rules.append(("A1", "nsubj", "S1"))
ALL_POS_EVENTUALITY_RULES["s-be-a"] = E_rule
NOMINAL_SAVED_COMPONENTS["s-be-a"] = {"S1"}
VERBAL_SAVED_COMPONENTS["s-be-a"] = set()


# subj-be-obj
E_rule = EventualityRule()
E_rule.positive_rules.append(("A1", "^cop", "V1"))
E_rule.positive_rules.append(("A1", "nsubj", "S1"))
ALL_POS_EVENTUALITY_RULES["s-be-o"] = E_rule
NOMINAL_SAVED_COMPONENTS["s-be-o"] = {"S1", "A1"}
VERBAL_SAVED_COMPONENTS["s-be-o"] = set()


# subj-verb-be-object
E_rule = EventualityRule()
E_rule.positive_rules.append(("V1", "nsubj", "S1"))
E_rule.positive_rules.append(("V1", "xcomp", "A1"))
E_rule.positive_rules.append(("A1", "cop", "V2"))
ALL_POS_EVENTUALITY_RULES["s-v-be-a"] = E_rule
ALL_POS_EVENTUALITY_RULES["s-v-be-o"] = E_rule
NOMINAL_SAVED_COMPONENTS["s-v-be-a"] = {"S1"}
NOMINAL_SAVED_COMPONENTS["s-v-be-o"] = {"S1", "A1"}
VERBAL_SAVED_COMPONENTS["s-v-be-a"] = {"V1"}
VERBAL_SAVED_COMPONENTS["s-v-be-o"] = {"V1"}


# nsubjpass-verb
E_rule = EventualityRule()
E_rule.positive_rules.append(("V1", "nsubjpass", "S1"))
ALL_POS_EVENTUALITY_RULES["spass-v"] = E_rule
NOMINAL_SAVED_COMPONENTS["spass-v"] = {"S1"}
VERBAL_SAVED_COMPONENTS["spass-v"] = {"V1"}


# nsubj-verb-dobj1-verb-dobj2
E_rule = EventualityRule()
E_rule.positive_rules.append(("V1", "nsubj", "S1"))
E_rule.positive_rules.append(("V1", "dobj", "O1"))
E_rule.positive_rules.append(("V1", "xcomp", "A1"))
E_rule.positive_rules.append(("A1", "cop", "V2"))
ALL_POS_EVENTUALITY_RULES["s-v-o-be-a"] = E_rule
ALL_POS_EVENTUALITY_RULES["s-v-o-be-o"] = E_rule
NOMINAL_SAVED_COMPONENTS["s-v-o-be-a"] = {"S1", "O1"}
NOMINAL_SAVED_COMPONENTS["s-v-o-be-o"] = {"S1", "O1", "A1"}
VERBAL_SAVED_COMPONENTS["s-v-o-be-a"] = {"V1"}
VERBAL_SAVED_COMPONENTS["s-v-o-be-o"] = {"V1"}

# nsbjpass-verb-verb-dobj
E_rule = EventualityRule()
E_rule.positive_rules.append(("V1", "nsubjpass", "S1"))
E_rule.positive_rules.append(("V1", "xcomp", "V2"))
E_rule.positive_rules.append(("V2", "dobj", "O1"))
ALL_POS_EVENTUALITY_RULES["spass-v-v-o"] = E_rule
NOMINAL_SAVED_COMPONENTS["spass-v-v-o"] = {"S1", "O1"}
VERBAL_SAVED_COMPONENTS["spass-v-v-o"] = {"V1", "V2"}

# nsubj-verb-dobj
E_rule = EventualityRule()
E_rule.positive_rules.append(("V1", "nsubjpass", "S1"))
E_rule.positive_rules.append(("V1", "dobj", "O1"))
ALL_POS_EVENTUALITY_RULES["spass-v-o"] = E_rule
NOMINAL_SAVED_COMPONENTS["spass-v-o"] = {"S1", "O1"}
VERBAL_SAVED_COMPONENTS["spass-v-o"] = {"V1"}

# there-be-obj
E_rule = EventualityRule()
E_rule.positive_rules.append(("V1", "nsubj", "S1"))
E_rule.positive_rules.append(("V1", "expl", "ex1"))
ALL_POS_EVENTUALITY_RULES["there-be-o"] = E_rule
NOMINAL_SAVED_COMPONENTS["there-be-o"] = {"S1"}
VERBAL_SAVED_COMPONENTS["there-be-o"] = set()


if __name__ == "__main__":
    EVENTUALITY_PATTERNS = {"s-v", "s-v-o", "s-v-a", "s-v-o-o", "s-be-a", "s-be-o", "s-v-be-a", "s-v-be-o", "s-v-v-o",
                            "s-v-v", "spass-v", "s-v-o-v-o", "s-v-o-be-a", "s-v-o-be-o", "spass-v-v-o", "spass-v-o",
                            "there-be-o", "s-v-o-v-o-o"}
    covered_patterns = ALL_POS_EVENTUALITY_RULES.keys()
    print(EVENTUALITY_PATTERNS - covered_patterns)