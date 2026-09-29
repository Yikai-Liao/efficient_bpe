"""Independent prototype of live-boundary bitmap + block-indexed long skips.
Based on the layout idea in Bille/Goertz/Prezza's practical Re-Pair (2017).
No training code is changed. A permanent end sentinel simplifies lookup.
"""
from pathlib import Path as _AuditPath
REPO = _AuditPath(__file__).resolve().parents[2]
ARCHIVE = _AuditPath(__file__).resolve().parent

from array import array
import random
class Boundaries:
    def __init__(self,n):
        assert n<2**32
        self.n=n
        nbits=n+1 # includes end sentinel
        self.bits=array('Q',[(1<<64)-1])*((nbits+63)//64)
        self.bits[-1]=(1<<((nbits-1)%64+1))-1
        self.skips=array('I',[0])*len(self.bits)
    def alive(self,i):return bool(self.bits[i//64]>>(i%64)&1)
    def nxt(self,i):
        assert self.alive(i)
        if i==self.n:return None
        block,off=divmod(i,64)
        w=self.bits[block]>>(off+1)
        if w:return i+1+(w&-w).bit_length()-1
        w=self.bits[block+1]
        if w:return (block+1)*64+(w&-w).bit_length()-1
        return i+self.skips[block+1]+1
    def prev(self,i):
        assert self.alive(i)
        if i==0:return None
        block,off=divmod(i,64)
        w=self.bits[block]&((1<<off)-1)
        if w:return block*64+w.bit_length()-1
        w=self.bits[block-1]
        if w:return (block-1)*64+w.bit_length()-1
        return i-self.skips[block-1]-1
    def merge(self,i):
        j=self.nxt(i);assert j<self.n
        k=self.nxt(j)
        self.bits[j//64]&=~(1<<(j%64))
        b1,b3=i//64,k//64
        if b3>b1+1:
            self.skips[b1+1]=self.skips[b3-1]=k-i-1
    def bytes(self):return len(self.bits)*self.bits.itemsize+len(self.skips)*self.skips.itemsize

def verify(b,live):
 for k,p in enumerate(live):
    assert b.alive(p)
    assert b.prev(p)==(live[k-1] if k else None)
    assert b.nxt(p)==(live[k+1] if k+1<len(live) else None)
 assert sum(w.bit_count() for w in b.bits)==len(live)

rng=random.Random(982);count=0
for n in [2,63,64,65,127,128,129,255,256,257,1024,4096,65536]:
 for trial in range(5):
    b=Boundaries(n);live=list(range(n+1))
    while len(live)>2:
        k=rng.randrange(len(live)-2) if trial%3==0 else (0 if trial%3==1 else len(live)-3)
        b.merge(live[k]);del live[k+1];count+=1
        if n<=257 or count%1023==0:verify(b,live)
    verify(b,live)
    assert b.nxt(0)==n
print({'merges_checked':count,'max_length_checked':65536,'buffer_bytes_for_65536_chars_and_sentinel':Boundaries(65536).bytes(),'asymptotic_bytes_per_position':12/64})
