"""Copy known benchmark data read-only, validating before any writes."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('source',type=Path)
    args=p.parse_args()
    dest=Path(__file__).resolve().parent.parent/'data'
    manifest=json.loads((dest/'snapshot-manifest.json').read_text())
    checked=[]
    for row in manifest:
        payload=(args.source/row['snapshot']).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=row['sha256']:
            raise ValueError('source hash differs from recorded snapshot: '+row['snapshot'])
        checked.append((row['snapshot'],payload))
    for name,payload in checked:
        (dest/name).write_bytes(payload)
    print('Verified and copied',len(checked),'files; source was read-only.')

if __name__=='__main__':main()
