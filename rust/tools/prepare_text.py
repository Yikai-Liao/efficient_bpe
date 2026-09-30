"""Prepare one continuous UTF-8 piece; preserve whitespace and punctuation."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    text = raw.decode("utf-8")
    alphabet = sorted(set(text))
    ids = {char: index for index, char in enumerate(alphabet, 1)}
    metadata = {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw), "source_characters": len(text),
        "symbols": [None, *alphabet], "pretokenization": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A bounded JSON chunk avoids materializing a Python int per corpus position.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False,
                                     dir=args.output.parent, suffix=".prepared.tmp") as stream:
        temporary = Path(stream.name)
        try:
            stream.write('{"corpus":[0')
            for start in range(0, len(text), 65536):
                values = [ids[char] for char in text[start:start + 65536]]
                stream.write("," + json.dumps(values, separators=(",", ":"))[1:-1])
            if text:
                stream.write(",0")
            stream.write('],"initial_lengths":')
            json.dump([1] * (len(alphabet) + 1), stream)
            stream.write(',"pivots":' + ("[1]" if text else "[]"))
            stream.write(',"weights":' + ("[1]" if text else "[]"))
            stream.write(',"text_metadata":')
            json.dump(metadata, stream, ensure_ascii=False)
            stream.write("}\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "alphabet_size": len(alphabet),
                      **{key: value for key, value in metadata.items() if key != "symbols"}}))


if __name__ == "__main__":
    main()
