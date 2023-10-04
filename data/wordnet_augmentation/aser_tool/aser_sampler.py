import random

class Sampler:
    def __init__(self, G_aser):
        self.G_aser = G_aser

    def unified_sample(self, args):
        if args.sample == "random":
            node_list = self.sample_randomly(args.number, args.pattern)
        elif args.sample == "gpt2":
            node_list = self.sample_by_gpt2(args.number, args.pattern, args.start_idx)
        elif args.sample == "name":
            node_list = self.sample_by_name(args.name)
        elif args.sample == "keyword":
            node_list = self.sample_by_keyword(args.number, args.key_word, args.start_idx)
        else:
            raise ValueError("Wrong value of args.sample: {}".format(args.sample))
        return node_list

    def sample_randomly(self, number, pattern):
        nodes = list([node for node in self.G_aser.nodes if self.G_aser.nodes[node]["info"]["pattern"] == pattern])
        return random.sample(nodes, k=min(number, len(nodes)))

    def sample_by_gpt2(self, number, pattern, start_idx):
        node_list = [node for node in self.G_aser.nodes if pattern is None
                     or pattern == self.G_aser.nodes[node]["info"]["pattern"]]
        node_score_list = [self.G_aser.nodes[node]["gpt2"] for node in node_list]
        node_score_list, node_list = zip(*sorted(zip(node_score_list, node_list), key=lambda x: x[0]))
        return node_list[start_idx: start_idx + number]

    def sample_by_name(self, name):
        assert name in self.G_aser.nodes, "the node \"{}\" is not in the current ASER".format(name)
        node_list = [name]
        return node_list

    def sample_by_keyword(self, number, key_word, start_idx):
        node_list = [node for node in self.G_aser.nodes if key_word in node]
        node_score_list = [self.G_aser.nodes[node]["gpt2"] for node in node_list]
        node_score_list, node_list = zip(*sorted(zip(node_score_list, node_list), key=lambda x: x[0]))
        return node_list[start_idx: start_idx + number]
