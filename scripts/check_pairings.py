"""Check speaker pairing patterns."""
import sys
sys.path.insert(0, "src")
from research.config import Config
from research.dataset import load_all_metadata

cfg = Config()
meta = load_all_metadata(cfg)

print("=== Speaker Pairings ===")
for cid in sorted(meta):
    sa = meta[cid]["speakers"]["a"]["speaker_id"]
    sb = meta[cid]["speakers"]["b"]["speaker_id"]
    print(f"  {cid[:20]}  A={sa}  B={sb}  dur={meta[cid]['duration_seconds']:.0f}s")

print()
from collections import Counter
spk_conv_counts = Counter()
for cid, rec in meta.items():
    for side in ("a", "b"):
        spk_conv_counts[rec["speakers"][side]["speaker_id"]] += 1
print("Speaker appearances:", spk_conv_counts.most_common())
