"""
RAG Demo 配置模板 —— 复制为 config.py 并填入实际路径
"""
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# 知识库笔记目录
NOTES_DIR = r"D:\path\to\your\notes"

# Chroma 持久化存储目录
DB_DIR = r"D:\path\to\rag_db"

# 本地 Embedding 模型路径（bge-small-zh-v1.5 或类似中文模型）
EMBED_MODEL_PATH = r"D:\path\to\your\model"

embedding_func = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_PATH)
