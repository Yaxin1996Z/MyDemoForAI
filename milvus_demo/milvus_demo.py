"""
Milvus 向量数据库 Python Demo
────────────────────────────
覆盖场景：
  1. 连接 & 创建 Collection
  2. 写入向量数据（模拟 20 篇新闻嵌入）
  3. 构建 HNSW 索引
  4. 向量相似度搜索（Top-5）
  5. 混合搜索（向量 + 标量过滤）
  6. Attu 可视化提示
"""

import random
import sys
import time

from pymilvus import (
    MilvusClient,
    DataType,
    AnnSearchRequest,
    RRFRanker,
)

# ── 配置 ──────────────────────────────────────────────────────────────
# 两种模式：
#   1) Docker Compose 模式: MILVUS_URI = "http://localhost:19530"
#   2) Milvus Lite 模式:    MILVUS_URI = "./milvus_demo.db"  （无需 Docker）
MILVUS_URI = "http://localhost:19530"  # Docker 模式

COLLECTION_NAME = "news_demo"
DIM = 64  # 嵌入维度（demo 用小维度，生产通常是 768/1024）

SEPARATOR = "\n" + "=" * 72 + "\n"


# ── 第1步：模拟生成向量和新闻数据 ────────────────────────────────────
def generate_demo_data(n: int = 20):
    """生成 n 篇"新闻"，每篇带一个 64 维随机向量"""
    topics = [
        "AI Agent 框架对比：LangGraph vs CrewAI",
        "Redis 8.0 正式支持向量搜索功能",
        "Milvus 3.0 发布 Loon 存储引擎",
        "2026 年大模型趋势：多模态成为标配",
        "RAG 优化：混合检索与重排序实战",
        "HNSW 算法原理与参数调优指南",
        "Python 异步编程在 Agent 中的应用",
        "Docker 容器化部署 AI 应用最佳实践",
        "向量数据库选型：Milvus vs Qdrant vs pgvector",
        "Prompt Engineering 高级技巧：思维链",
        "从零搭建多 Agent 协作系统",
        "MCP 协议：AI Agent 的工具调用标准",
        "Function Calling 原理与 JSON Schema 设计",
        "LLM 评估体系：从单元测试到端到端评测",
        "AI Agent 安全护栏：防注入与权限控制",
        "SFT 与 LoRA 微调实战指南",
        "知识图谱 + RAG：GraphRAG 实践",
        "AI 应用的可观测性：LangSmith 入门",
        "边缘设备上的轻量级向量搜索",
        "2026 年 AI 工程师面试高频考点汇总",
    ]

    data = []
    for i, title in enumerate(topics):
        # 随机向量：模拟经过 embedding 模型转换后的语义向量
        vector = [random.random() for _ in range(DIM)]

        # 标量字段：模拟元数据
        word_count = random.randint(200, 3000)
        category = random.choice(["AI框架", "数据库", "大模型", "工程实践", "面试"])

        data.append({
            "id": i + 1,
            "title": title,
            "category": category,
            "word_count": word_count,
            "embedding": vector,
        })

    return data


# ── 第2步：创建 Collection ────────────────────────────────────────────
def create_collection(client: MilvusClient):
    """创建 Collection（类比 SQL 的 CREATE TABLE）"""
    print(f"  创建 Collection: {COLLECTION_NAME}")

    # 如果已存在则删除（方便重复运行）
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print("  已删除旧 Collection")

    # 定义 Schema（表的字段结构）
    schema = MilvusClient.create_schema(
        auto_id=False,          # 手动指定 id
        enable_dynamic_field=False,
    )

    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("title", DataType.VARCHAR, max_length=128)
    schema.add_field("category", DataType.VARCHAR, max_length=32)
    schema.add_field("word_count", DataType.INT64)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIM)

    # 创建 Collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
    )
    print(f"  Collection 创建成功 (dim={DIM})")


# ── 第3步：构建索引 ──────────────────────────────────────────────────
def create_index(client: MilvusClient):
    """在 embedding 字段上创建 HNSW 索引（建索引后才能搜索）"""
    print(f"  构建 HNSW 索引...")

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",              # 最主流的 ANN 算法
        metric_type="IP",               # 内积距离（也可以用 COSINE / L2）
        params={
            "M": 16,                    # 每个节点的最大连接数（越大召回越高，内存越大）
            "efConstruction": 200,      # 构建时的动态候选数
        },
    )

    client.create_index(
        collection_name=COLLECTION_NAME,
        index_params=index_params,
    )
    print(f"  HNSW 索引创建完成 (M=16, efConstruction=200)")


# ── 第4步：写入数据 ──────────────────────────────────────────────────
def insert_data(client: MilvusClient, data: list):
    """插入新闻数据"""
    print(f"  插入 {len(data)} 条数据...")
    res = client.insert(collection_name=COLLECTION_NAME, data=data)
    print(f"  插入成功，影响行数: {res['insert_count']}")


# ── 第5步：向量搜索 ──────────────────────────────────────────────────
def search_similar(client: MilvusClient, query_vector: list, top_k: int = 5):
    """基础向量搜索：找最相似的 top_k 条"""
    # 需要先加载 Collection 到内存
    client.load_collection(COLLECTION_NAME)

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="embedding",
        limit=top_k,
        output_fields=["title", "category", "word_count"],
    )

    return results[0]


# ── 第6步：混合搜索（向量 + 标量过滤） ────────────────────────────────
def search_with_filter(
    client: MilvusClient,
    query_vector: list,
    category_filter: str,
    top_k: int = 5,
):
    """混合搜索：向量相似度 + 按分类过滤"""
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="embedding",
        limit=top_k,
        output_fields=["title", "category", "word_count"],
        # 关键：标量过滤表达式
        filter=f'category == "{category_filter}"',
    )
    return results[0]


# ── 第7步：打印结果 ──────────────────────────────────────────────────
def print_results(results, label: str):
    print(f"\n  【{label}】")
    for i, hit in enumerate(results):
        entity = hit["entity"]
        print(f"    #{i + 1}  score={hit['distance']:.4f}")
        print(f"        标题: {entity['title']}")
        print(f"        分类: {entity['category']}  |  字数: {entity['word_count']}")


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════
def main():
    print(SEPARATOR)
    print("【Milvus 向量数据库 Demo】")
    print(f"  服务器: {MILVUS_URI}")
    print(SEPARATOR)

    # ── 连接 Milvus ────────────────────────────────────────────────
    print("第1步：连接 Milvus（Docker Compose 模式）")
    client = MilvusClient(uri=MILVUS_URI)
    version = client.get_server_version()
    print(f"  ✓ 连接成功 (Milvus 版本: {version})")
    print(SEPARATOR)

    # ── 创建 Collection + 索引 ─────────────────────────────────────
    print("第2步：创建 Collection & 索引")
    create_collection(client)
    create_index(client)
    print(SEPARATOR)

    # ── 插入数据 ───────────────────────────────────────────────────
    print("第3步：插入模拟新闻数据")
    data = generate_demo_data(20)
    insert_data(client, data)
    print(SEPARATOR)

    # ── 等待索引生效 ───────────────────────────────────────────────
    print("第4步：等待索引构建完成（等 2 秒）")
    client.load_collection(COLLECTION_NAME)
    time.sleep(2)
    print(SEPARATOR)

    # ── 向量搜索 ───────────────────────────────────────────────────
    print("第5步：向量相似度搜索（找最相似的 5 篇）")

    # 用第 1 篇新闻的向量作为查询
    query_vector = data[0]["embedding"]
    print(f"  查询向量来自: {data[0]['title']}\n")

    results = search_similar(client, query_vector, top_k=5)
    print_results(results, "与查询最相似的 5 篇新闻")
    print(SEPARATOR)

    # ── 混合搜索 ───────────────────────────────────────────────────
    print("第6步：混合搜索（仅限 '数据库' 分类）")
    results_filtered = search_with_filter(
        client, query_vector, category_filter="数据库", top_k=3
    )
    print_results(results_filtered, "仅「数据库」分类中最相似的 3 篇")
    print(SEPARATOR)

    # ── 清理（可选）───────────────────────────────────────────────
    print("第7步：清理")
    # 取消注释下一行即可删除 Collection（清空所有数据）
    # client.drop_collection(COLLECTION_NAME)
    # print("  Collection 已删除")
    print("  数据保留，可继续查询。如需删除，取消上面注释即可。")
    print(SEPARATOR)

    # ── 说明 ─────────────────────────────────────────────────────
    print("【说明】")
    print("  本 Demo 连接远端 Milvus 服务器（Docker Compose）")
    print("  如需切换为 Lite 模式，改 uri 为:")
    print('    MILVUS_URI = "./milvus_demo.db"')
    print("  可视化工具 Attu（可选）:")
    print("  docker run -d -p 8000:8000 --name milvus-attu \\")
    print("    -e MILVUS_URL=localhost:19530 zilliz/attu:latest")
    print(SEPARATOR)

    print("✓ Demo 运行完毕")
    return 0


if __name__ == "__main__":
    sys.exit(main())
