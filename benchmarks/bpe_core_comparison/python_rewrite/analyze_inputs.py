import json
from pathlib import Path
from bench import load_pieces
from common import prepare

rows=[]
for name in ['en-1m','zh-1m','de-1m','ja-1m','runs-131072']:
    pieces, byte_count, digest=load_pieces(name,'regex')
    corpus,lengths,pivots,weights=prepare(pieces)
    n_chars=n_words=n_runs=n_selfruns=0
    last=0
    run_length=0
    for token in corpus:
        if token==0:
            if last:
                n_words+=1
                if run_length>1:n_selfruns+=1
            last=0
            run_length=0
        else:
            n_chars+=1
            if token!=last:
                if run_length>1:n_selfruns+=1
                n_runs+=1
                last=token
                run_length=1
            else:
                run_length+=1
    yttm_occ=n_runs-n_words+n_selfruns
    row={'dataset':name,'physical_characters':n_chars,'stored_words':n_words,
         'original_positions_with_separators':len(corpus),'runs':n_runs,
         'self_runs':n_selfruns,'positions_per_run':n_chars/n_runs,
         'flat_initial_occurrences':n_chars-n_words,
         'yttm_initial_occurrences':yttm_occ,'frequency_groups':len(pivots),
         'yttm_node_payload_bytes':16*n_runs,'yttm_occ_payload_bytes':16*yttm_occ,
         'flat_v2_corpus_payload_bytes':4*len(corpus),
         'flat_u32_occ_payload_bytes':4*(n_chars-n_words),
         'flat_u64_occ_payload_bytes':8*(n_chars-n_words)}
    rows.append(row)
path=Path(__file__).resolve().parent.parent/'input-layout-counts.json'
path.write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
