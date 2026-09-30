import importlib.util
from pathlib import Path

OUT = Path(__file__).resolve().parent
source = OUT.parent / "radical-adaptive-replay-v1/summarize.py"
spec = importlib.util.spec_from_file_location("adaptive_replay_summary", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.summarize(OUT)
