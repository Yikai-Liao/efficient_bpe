"""Queue comparison with identical fused endpoints and identical semantics."""
import argparse
import gc
import json
import resource
from bench import load_pieces
from common import prepare
from common_queue import train
from backends_fused import FusedEbpeEndpoints
from queues import HeapQueue, HighLowQueue

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True)
    p.add_argument('--queue',choices=['heap','bucket'],required=True)
    p.add_argument('--rules',type=int,default=3000)
    p.add_argument('--weight-scale',type=int,default=1)
    a=p.parse_args()
    pieces,nbytes,digest=load_pieces(a.dataset,'regex')
    prepared=prepare(pieces)
    prepared[3][:]=[w*a.weight_scale for w in prepared[3]]
    del pieces
    gc.collect()
    cls=HeapQueue if a.queue=='heap' else HighLowQueue
    result=train(prepared,FusedEbpeEndpoints,a.rules,min_frequency=2*a.weight_scale,queue_class=cls)
    result.update(dataset=a.dataset,queue=a.queue,weight_scale=a.weight_scale,
                  backend='endpoints',split='regex',deduplicate=True,
                  requested_rules=a.rules,input_bytes=nbytes,input_sha256=digest,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    print(json.dumps(result))

if __name__=='__main__':main()
