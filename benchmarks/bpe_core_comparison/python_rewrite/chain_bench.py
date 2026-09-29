"""A legal greedy-BPE chain: descending distinct IDs, one weighted piece.

All current pairs have frequency 2. Lexicographic pair ties select the
rightmost lowest left-ID pair, then repeatedly prepend its predecessor.
This tests the full greedy loop, not an arbitrary forced merge schedule.
"""
import argparse
import json
import resource
from common import prepare
from common_fused import train
from backends_fused import FusedEbpeEndpoints, FusedPrezzaHalfword
from backend_linked_fused import FusedLinked12

class FullClear(FusedEbpeEndpoints):
    def merge_known(self,pos,right,after,new_id,new_len):
        corpus=self.corpus
        corpus[pos]=corpus[after-1]=new_id
        for i in range(pos+1,after-1):
            corpus[i]=0
        return right

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--length',type=int,required=True)
    p.add_argument('--backend',required=True)
    a=p.parse_args()
    word=''.join(chr(0x1000+i) for i in range(a.length,0,-1))
    inp=prepare([word,word])
    cls={'full_clear':FullClear,'endpoints':FusedEbpeEndpoints,
         'linked12':FusedLinked12,'bitmap_halfword':FusedPrezzaHalfword}[a.backend]
    result=train(inp,cls,a.length,min_frequency=2)
    assert result['actual_merges']==result['rules']==a.length-1
    assert result['max_token_length']==a.length
    result.update(backend=a.backend,length=a.length,dataset='descending-id-chain',
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    print(json.dumps(result))

if __name__=='__main__':main()
