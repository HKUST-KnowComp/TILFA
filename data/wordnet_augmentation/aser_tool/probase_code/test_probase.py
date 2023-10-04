import json
import copy
from threading import Lock

top_concepts = ['factor', 'feature', 'issue', 'topic', 'name', 'aspect', 'concept', 'entity', 'component', 'reason',
                'characteristic', 'theme', 'category', 'form', 'alternative', 'part', 'type', 'attribute', 'class',
                'instance', 'keyword', 'suffix', 'ambiguous reference']


class ProbaseClient:
    def __init__(self):
        from multiprocessing.connection import Client
        address = ('localhost', 9623)
        print('Connecting Probase...')
        self.conn = Client(address)
        print('Connected')
        self.cache = {}
        self._lock = Lock()

    def query(self, x, sort_method='mi', truncate=10):
        x = x.lower()
        if (x, sort_method, truncate) in self.cache:
            return copy.copy(self.cache[(x, sort_method, truncate)])
        with self._lock:
            self.conn.send(json.dumps([x, sort_method, truncate]))
            res = json.loads(self.conn.recv())
            key_remove = []
            for key in res:
                if key.split(' ')[-1] in ['word', 'phrase', 'noun', 'adjective', 'verb', 'pronoun', 'term',
                                          'aux'] or key in top_concepts:
                    key_remove.append(key)
            for k in key_remove:
                del res[k]
            self.cache[(x, sort_method, truncate)] = copy.copy(res)
        return res


pb = ProbaseClient()

res = pb.query("analyst")
print(res)
