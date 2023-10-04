class ConceptResult:
    def __init__(self, synset_list, instance, start_idx, end_idx,
                 head_idx, source, event_text, token_list):
        self.synset_list = synset_list
        self.instance = instance
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.head_idx = head_idx
        self.source = source
        self.event_text = event_text
        self.token_list = token_list
        self.example_id = 0
        assert self.source in {"wn", "pb", "root phrase", "none"}

    def __str__(self):
        return "{}, {}, {}, {}".format(self.instance, self.start_idx, self.end_idx, self.source)

    @staticmethod
    def load_from_dict(var):
        instance, start_idx, end_idx = var["instance"], var["start_idx"], var["end_idx"]
        head_idx, event_text, token_list = var["head_idx"], var["event_text"], var["token_list"]
        synset_list = var["synset_list"] if "synset_list" in var else None
        source = var["source"] if "source" in var else "none"
        res = ConceptResult(synset_list=synset_list, instance=instance, start_idx=start_idx, end_idx=end_idx,
                            head_idx=head_idx, source=source, event_text=event_text, token_list=token_list)
        return res

    def save_to_dict(self,):
        res_dict = {"event_text": self.event_text, "instance": self.instance,
                    "start_idx": self.start_idx, "end_idx": self.end_idx, "head_idx": self.head_idx,
                    "token_list": self.token_list}
        return res_dict

    def save_to_dict_with_synset_source(self):
        res_dict = self.save_to_dict()
        res_dict["synset_list"] = self.synset_list
        res_dict["source"] = self.source
        return res_dict

