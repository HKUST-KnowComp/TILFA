from multiprocessing.connection import Listener
import json
from conceptualize_proposer import Proposer
from build_graph_utils import pb_query_abstract
import traceback
import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--probase_path", type=str,
                    default="/home/data/zwanggy/kg_data/probase/data-concept-instance-relations.txt")
args = parser.parse_args()

print('Starting server...')
proposer = Proposer(args.probase_path)

address = ('localhost', 9623)     # family is deduced to be 'AF_INET'
while True:
    listener = Listener(address)
    print('Server started')
    conn = listener.accept()
    print('Connection accepted from', listener.last_accepted)
    while True:
        try:
            msg = conn.recv()
            if msg is None:
                break
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M '), msg)
            if not isinstance(msg, str):
                conn.send('')
                continue
            msg = json.loads(msg)
            response = pb_query_abstract(proposer, msg[0])
            print('Collected probase response of %d items' % len(response))
            if msg[1] != '':
                response = list(response.items())
                response.sort(key=lambda x: -x[1][msg[1]])
                if msg[2] != -1:
                    response = response[:msg[2]]
                # response = dict(response)
            response = json.dumps(response)
            print('Send %d bytes' % len(response))
            conn.send(response)
        except:
            traceback.print_exc()
            break

    listener.close()
