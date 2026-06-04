"""
RAG Demo —— 个人知识库问答

数据源：技能面板下的学习笔记（Markdown）
向量库：Chroma（持久化存储）
流程：读取笔记 → 分块 → Embedding → 检索 → LLM 生成

用法：
  python rag_demo.py              # 首次运行：建库 + 问答
  python rag_demo.py --rebuild    # 强制重建索引
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import os
import glob
import re

import chromadb
from openai import OpenAI

from config import NOTES_DIR, DB_DIR, embedding_func

# ============================================================
# 1. 读取 Markdown 笔记 → 分块
# ============================================================
def read_markdown_files(directory: str) -> list[dict]:
    """读取目录下所有 .md 文件，返回 [{file, title, content}, ...]"""
    files = glob.glob(os.path.join(directory, "*.md"))
    # 排除 skill-data.js 之类非笔记文件
    files = [f for f in files if os.path.basename(f) != "skill-data.js"]

    docs = []
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        with open(fp, "r", encoding="utf-8") as f:
            content = f.read()
        docs.append({"file": name, "title": name, "content": content})
    return docs


def chunk_document(doc: dict) -> list[dict]:
    """
    按 ## 章节标题切分文档。
    返回 [{id, file, chapter, text, metadata}, ...]
    """
    content = doc["content"]
    file_name = doc["file"]

    # 按 ## 二级标题分割
    sections = re.split(r"\n(?=## )", content)

    chunks = []
    for i, section in enumerate(sections):
        lines = section.strip().split("\n")
        # 提取标题行
        chapter = ""
        for line in lines:
            if line.startswith("## "):
                chapter = line.replace("## ", "").strip()
                break

        # 跳过学习进度/元信息等短段落
        text = section.strip()
        if len(text) < 20:
            continue

        chunk_id = f"{file_name}_{i}"
        chunks.append({
            "id": chunk_id,
            "file": file_name,
            "chapter": chapter or file_name,
            "text": text,
            "metadata": {
                "source": file_name,
                "chapter": chapter or file_name,
                "chunk_index": i,
            },
        })

    return chunks


# ============================================================
# 2. 构建 Chroma 索引
# ============================================================
def build_index(force_rebuild: bool = False):
    """检查索引状态，不存在或 --rebuild 时才读取文件建库"""
    client = chromadb.PersistentClient(path=DB_DIR)

    # 有索引且不强制重建 → 直接返回
    if not force_rebuild:
        try:
            collection = client.get_collection("skill_notes")
            print(f"  📦 使用已有索引（共 {collection.count()} 条）")
            return collection
        except (ValueError, chromadb.errors.NotFoundError):
            pass

    # 需要重建：读取文件 → 分块 → 写入
    if force_rebuild:
        try:
            client.delete_collection("skill_notes")
        except:
            pass

    collection = client.create_collection(name="skill_notes", embedding_function=embedding_func)

    docs = read_markdown_files(NOTES_DIR)
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunk_document(d))

    print(f"  📄 共读取 {len(docs)} 篇笔记，切分为 {len(all_chunks)} 个块")

    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        collection.add(
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
            ids=[c["id"] for c in batch],
        )

    print(f"  ✅ 已写入 {collection.count()} 条到 Chroma")
    return collection


# ============================================================
# 3. 检索 + 生成
# ============================================================
def rag_query(collection, question: str, llm_client: OpenAI, k: int = 3):
    """检索 Top-K + LLM 生成回答"""
    # ---------- 检索 ----------
    results = collection.query(
        query_texts=[question],
        n_results=k,
    )

    retrieved_docs = results["documents"][0]
    retrieved_metas = results["metadatas"][0]
    retrieved_distances = results["distances"][0] if results["distances"] else []

    # ---------- 组装上下文 ----------
    context_parts = []
    for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas)):
        source = meta.get("source", "?")
        chapter = meta.get("chapter", "?")
        context_parts.append(
            f"【来源 {i+1}】{source} / {chapter}\n{doc}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # ---------- 生成 ----------
    system_prompt = """你是一个基于知识库回答问题的助手。

【规则】
- 只基于下面的"参考资料"来回答
- 如果参考资料不足以回答问题，说"我的知识库中没有相关信息"
- 不要编造答案
- 在回答末尾标注信息来源（来源文件名）"""

    user_prompt = f"""【参考资料】
{context}

【问题】
{question}

【回答】"""

    response = llm_client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    answer = response.choices[0].message.content

    return {
        "question": question,
        "answer": answer,
        "retrieved": [
            {
                "source": meta.get("source", "?"),
                "chapter": meta.get("chapter", "?"),
                "distance": dist,
                "preview": doc[:80] + "...",
            }
            for doc, meta, dist in zip(
                retrieved_docs, retrieved_metas, retrieved_distances
            )
        ],
    }


# ============================================================
# 4. Demo 运行
# ============================================================
def print_result(result: dict):
    print(f"\n{'='*60}")
    print(f"[问题] {result['question']}")
    print(f"{'='*60}")
    print(f"[回答] {result['answer']}")
    print(f"\n[引用资料]")
    for r in result["retrieved"]:
        dist_text = f"  📖 {r['source']} → {r['chapter']}"
        if r["distance"] is not None:
            dist_text += f"  (相似度: {1 - r['distance']:.3f})"
        print(dist_text)
    print()


if __name__ == "__main__":
    force_rebuild = "--rebuild" in sys.argv

    # 初始化 LLM
    llm = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    # ---------- 建库 ----------
    print(f"\n📚 知识库: {NOTES_DIR}")
    collection = build_index(force_rebuild=force_rebuild)
    print(f"  ✅ 就绪\n")

    # ---------- 问答测试 ----------
    test_questions = [
        "LangChain 四大核心抽象是什么？",
        "Chain 和 Agent 的区别是什么？",
        "ReAct 循环的工作流程是怎样的？",
        "什么是 Function Calling？",
        "Prompt Engineering 有哪些核心技巧？",
        "什么是 RAG？",
        "今天天气怎么样？",  # 知识库里没有 → 测试兜底
    ]

    for q in test_questions:
        result = rag_query(collection, q, llm)
        print_result(result)
