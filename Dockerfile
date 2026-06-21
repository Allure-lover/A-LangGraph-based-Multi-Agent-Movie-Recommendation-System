# LangGraph Recommender System — Backend Docker Image
# Build context: project root (langgraph_study/)
FROM python:3.12-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖（从项目根目录复制）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY backend/ .

# 创建数据目录（挂载点）
RUN mkdir -p /app/data

# 暴露端口
EXPOSE 8000
EXPOSE 8501

# 默认启动 API Server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
