from pathlib import Path as _AuditPath
REPO = _AuditPath(__file__).resolve().parents[2]
ARCHIVE = _AuditPath(__file__).resolve().parent
import pathlib,inspect,random,collections,sys
sys.path.insert(0,str(REPO))
import ebpe_v2 as v2
base=pathlib.Path(str(ARCHIVE / 'checks.py')).read_text().split("print('V1_LONG_TOKEN')")[0]
ns={};exec(base,ns)
source=inspect.getsource(v2.merge_token_pair)
source=source.replace('        corpus[pos_x] = corpus[pos_end - 1] = new_token\n        for i in range(pos_x + 1, pos_end - 1):\n            corpus[i] = 0','        corpus[pos_y - 1] = corpus[pos_y] = 0\n        corpus[pos_x] = corpus[pos_end - 1] = new_token')
source=source.replace('        if l > 1 and len_pair + len_l <= max_piece_length:\n            # not <PAD> or <UNK>\n            pair_freq_patch[(l, x)] -= freq','        if l > 1:\n            pair_freq_patch[(l, x)] -= freq\n        if l > 1 and len_pair + len_l <= max_piece_length:')
source=source.replace('        if r > 1 and len_pair + len_r <= max_piece_length:\n            # not <PAD> or <UNK>\n            pair_freq_patch[(y, r)] -= freq','        if r > 1:\n            pair_freq_patch[(y, r)] -= freq\n        if r > 1 and len_pair + len_r <= max_piece_length:')
module=dict(vars(v2));exec(source,module);v2.merge_token_pair=module['merge_token_pair']
rng=random.Random(3829)
for maxlen in [2,3,4,8,100000]:
 for _ in range(1000):
  words=collections.Counter({''.join(rng.choices('abc',k=rng.randint(2,18))):rng.randint(1,8) for j in range(rng.randint(1,10))})
  err=ns['check'](words,maxlen,rng.randint(1,5))
  if err:print('FAIL',maxlen,words,err);raise SystemExit(1)
print('V2 scratch boundary-write + unconditional old-pair decrement passed 5000 random corpora')
pathlib.Path(str(ARCHIVE / 'v2_merge_scratch.py')).write_text('from collections import defaultdict\nimport bisect\n'+source)
