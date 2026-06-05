"""
查看 Chroma 知识库内容
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import chromadb
from config import DB_DIR

client = chromadb.PersistentClient(path=DB_DIR)
col = client.get_collection("skill_notes")

all_data = col.get()
print(f"\n{'='*50}")
print(f"  知识库总览")
print(f"{'='*50}")
print(f"  集合: skill_notes")
print(f"  总数: {col.count()} 条")
print(f"  来源: {set(m['source'] for m in all_data['metadatas'])}")
print(f"{'='*50}\n")

for i, (doc, meta, idx) in enumerate(zip(all_data["documents"], all_data["metadatas"], all_data["ids"])):
    print(f"  [{i:02d}] {meta['source']} / {meta['chapter']}")
    print(f"       {idx}")
    print(f"       {doc[:80].replace(chr(10), ' ')}...")
    print()
