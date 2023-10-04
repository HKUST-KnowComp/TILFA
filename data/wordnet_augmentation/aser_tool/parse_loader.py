from .match_util import WordTypeMixin


class Token:
    def __init__(self, text, tag, idx, offset, dep_rel=None,
                 head=None, head_idx=None, children=None, children_idx=None,
                 is_nominal=None, is_predicate=None, is_prep=None, is_modifier=None):
        self.text = text
        self.tag = tag
        self.dep_rel = dep_rel
        self.head = head
        self.head_idx = head_idx
        self.children = [] if children is None else children
        self.children_idx = [] if children_idx is None else children_idx
        self.idx = idx  # idx is the token index for a word in a sentence
        self.offset = offset  # offset is the char index for a word in a sentence
        self.is_nominal = is_nominal
        self.is_predicate = is_predicate
        self.is_prep = is_prep
        self.is_modifier = is_modifier

    def __str__(self):
        return self.text


class ParsedEventuality:
    def __init__(self, pattern, event_text, token_list):
        self.pattern = pattern
        self.event_text = event_text
        self.token_list = token_list
        self.text_token_list = [token.text for token in token_list]

class ParseLoader(WordTypeMixin):
    def __init__(self, G_aser):
        self.G_aser = G_aser
        super().__init__()

    # parse those nodes
    def parse_nodes(self, logger, event_list):
        parsed_eventuality_list = []
        for event in event_list:
            text_tokens, token_obj_list = event.split(), []
            node_info = self.G_aser.nodes[event]['info']
            dep_list, tag_list = node_info["_dependencies"], node_info["pos_tags"]
            cur_offset = 0

            for idx, tk, tg in zip(range(len(text_tokens)), text_tokens, tag_list):
                token_obj_list.append(Token(tk, tg, idx, cur_offset))
                cur_offset += len(tk) + 1  # len(tk) + 1 takes a space between two words into consideration

            for head_idx, rel, dependent_idx in dep_list:
                head, dependent = token_obj_list[head_idx], token_obj_list[dependent_idx]
                # add head
                dependent.head_idx = head_idx
                dependent.head = head
                dependent.dep_rel = rel
                # add children
                head.children.append(dependent)
                head.children_idx.append(dependent_idx)

            for idx, token in enumerate(token_obj_list):
                if token.head is None:
                    token.head, token.head_idx, token.dep_rel = token, idx, "ROOT"

            for token in token_obj_list:
                token.is_nominal = self.is_possibly_nominal(token)
                token.is_predicate = self.is_possibly_predicate(token)
                token.is_prep = self.is_possibly_prep(token)
                token.is_modifier = self.is_modifier(token)
            pattern = node_info["pattern"]
            parsed_eventuality = ParsedEventuality(pattern, event, token_obj_list)
            parsed_eventuality_list.append(parsed_eventuality)
        # filter nodes by some rules
        parsed_eventuality_list = self.__filter_nodes(logger, parsed_eventuality_list)
        return parsed_eventuality_list

    # filter invalid nodes
    # TODO checking all filtering rules should be done
    def __filter_nodes(self, logger, parsed_eventuality_list):
        remained_node_list, failed_node_count = [], 0
        for i, parsed_event in enumerate(parsed_eventuality_list):
            roots = [t for t in parsed_event.token_list if t.dep_rel == "ROOT"]
            try:
                if len(roots) != 1:
                    raise ValueError("{} has more than one ROOT".format(parsed_event.event_text))
                for token in parsed_event.token_list:
                    if token.dep_rel == 'compound' and \
                            token.head.tag not in {"NN", "NNP", "NNS", "NNPS", 'CD', "VBG"}:
                        raise ValueError("{} has invalid compound: {}".format(parsed_event.event_text, token.text))
                remained_node_list.append(parsed_event)
            except ValueError as e:
                logger.info("Fail case: {}".format(str(e)))
                failed_node_count += 1

        logger.info("failed nodes: {}, total nodes: {}, ratio of failed notes: {}".format(
            failed_node_count, len(parsed_eventuality_list), failed_node_count / len(parsed_eventuality_list)))

        return remained_node_list
