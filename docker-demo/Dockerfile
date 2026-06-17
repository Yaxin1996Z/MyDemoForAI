# ====================================
# Docker 基础演示
# 从零构建一个 Python Web 服务镜像
# ====================================

# 1. FROM —— 选一个轻量基础镜像（阿里云镜像加速）
FROM registry.aliyuncs.com/apecloud/python:3.11-slim

# 2. WORKDIR —— 设置容器内工作目录
WORKDIR /app

# 3. COPY —— 把 app.py 复制进镜像
COPY app.py .

# 4. EXPOSE —— 声明容器端口（仅声明，不映射）
EXPOSE 8000

# 5. CMD —— 容器启动时执行的命令
CMD ["python", "app.py"]
