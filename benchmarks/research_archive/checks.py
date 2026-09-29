from pathlib import Path as _AuditPath
REPO = _AuditPath(__file__).resolve().parents[2]
ARCHIVE = _AuditPath(__file__).resolve().parent
import sys,io,contextlib,random,collections,traceback
sys.path.insert(0,str(REPO))
import ebpe,ebpe_v2 as v2

def check(words, maxlen=100000, minfreq=1):
    c2i,i2c=v2.compute_alphabet(words,[],0,[])
    lengths=[1]*len(i2c)
    corpus,pivots,freqs=v2.build_corpus(words,c2i,False,False)
    seen=set(c2i.values())
    positions,counts,q=v2.count_token_pairs(corpus,pivots,freqs,minfreq)
    seqs=[([c2i[c] for c in s],f) for s,f in words.items()]
    for step in range(sum(map(len,words))):
        expected=collections.Counter()
        for seq,f in seqs:
            for x,y in zip(seq,seq[1:]):
                if lengths[x]+lengths[y]<=maxlen:expected[x,y]+=f
        expected={p:f for p,f in expected.items() if f>=minfreq}
        target=min(expected,key=lambda p:(-expected[p],p)) if expected else None
        pair,f=v2.most_frequent_combination(q,counts,positions,minfreq,corpus,lengths)
        if pair!=target or (pair is not None and f!=expected[pair]):
            return {'step':step,'actual':(pair,f),'expected':(target,expected.get(target)), 'decoded_actual':None if pair is None else tuple(i2c[x] for x in pair),'decoded_expected':None if target is None else tuple(i2c[x] for x in target)}
        if pair is None:return None
        new=v2.assign_token(pair,c2i,i2c,seen,lengths)
        patches=v2.merge_token_pair(corpus,pair,new,positions.pop(pair),pivots,freqs,lengths,maxlen)
        v2.apply_patch(q,counts,positions,*patches,minfreq)
        for k,(seq,freq) in enumerate(seqs):
            merged=[];i=0
            while i<len(seq):
                if i+1<len(seq) and (seq[i],seq[i+1])==pair:
                    merged.append(new);i+=2
                else:merged.append(seq[i]);i+=1
            seqs[k]=(merged,freq)
    raise AssertionError('did not terminate')

print('V1_LONG_TOKEN')
try:
    ebpe.BPETrainer(10000,min_freq=1,single_char=False).train_from_iter(['a'*512]*4)
except Exception as e:print(type(e).__name__, str(e))

rng=random.Random(937)
for maxlen in [2,3,4,8,100000]:
    failures=[]
    for _ in range(1000):
        words=collections.Counter({''.join(rng.choices('abc',k=rng.randint(2,12))):rng.randint(1,8) for j in range(rng.randint(1,8))})
        try:err=check(words,maxlen)
        except Exception as e:err={'exception':type(e).__name__,'message':str(e)}
        if err:
            failures.append((words,err))
            break
    print('V2',maxlen,'FIRST_FAILURE',failures)

print('SMALL_CASES')
for words in [{'abac':1},{'aba':1},{'abab':1},{'aaa':1},{'aaaa':1},{'abca':2}]:
    try:print(words,check(words,2))
    except Exception as e:print(words,type(e).__name__,str(e))
