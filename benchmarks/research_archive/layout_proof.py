"""Scratch proof: one byte per original position; no overflow side table."""
from pathlib import Path as _AuditPath
REPO = _AuditPath(__file__).resolve().parents[2]
ARCHIVE = _AuditPath(__file__).resolve().parent

import random
class Layout:
    def __init__(self,n):self.a=bytearray(b'\1')*n
    def start_len(self,s):
        x=self.a[s]
        if 1<=x<=63:return x
        if x==126:return sum((self.a[s+1+j]&127)<<(7*j) for j in range(5))
        return 0
    def end_len(self,e):
        x=self.a[e]
        if x==1:return 1
        if 64<=x<=125:return x-62
        if x==127:return sum((self.a[e-1-j]&127)<<(7*j) for j in range(5))
        return 0
    def merge(self,s,left,right):
        # Invalidate the two absorbed endpoints before writing the new ones.
        self.a[s+left-1]=0
        self.a[s+left]=0
        length=left+right;e=s+length-1
        assert length<=0xffffffff
        if length<=63:
            self.a[s]=length;self.a[e]=length+62
        else:
            self.a[s]=126;self.a[e]=127
            for j in range(5):
                digit=128|((length>>(7*j))&127)
                self.a[s+1+j]=digit
                self.a[e-1-j]=digit

def verify(layout,lengths):
    starts={};ends={};p=0
    for length in lengths:
        starts[p]=length;ends[p+length-1]=length;p+=length
    assert p==len(layout.a)
    for pos in range(p):
        assert layout.start_len(pos)==starts.get(pos,0),(pos,'start')
        assert layout.end_len(pos)==ends.get(pos,0),(pos,'end')

rng=random.Random(829)
merges=0
for n in [2,3,63,64,127,128,254,255,256,257,511,512,1024,4096]:
 for trial in range(8):
    a=Layout(n);lengths=[1]*n
    while len(lengths)>1:
        k=rng.randrange(len(lengths)-1) if trial%3==0 else (0 if trial%3==1 else len(lengths)-2)
        s=sum(lengths[:k]);left,right=lengths[k:k+2]
        a.merge(s,left,right);lengths[k:k+2]=[left+right];merges+=1
        if n<=512 or len(lengths)%127==0:verify(a,lengths)
    verify(a,lengths)
# Encoding checks for lengths not allocated as arrays.
for x in [64,255,256,65535,65536,2**31,2**32-1]:
 digits=[128|((x>>(7*j))&127) for j in range(5)]
 assert sum((digit&127)<<(7*j) for j,digit in enumerate(digits))==x
print({'random_and_directional_merges':merges,'maximum_test_buffer_bytes':4096,'layout_bytes_per_position':1,'uint32_codec_boundaries':'passed'})
