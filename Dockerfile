FROM python:3.11-slim

WORKDIR /app

RUN sed -i 's@http://deb.debian.org@http://mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list && \
    sed -i 's@http://security.debian.org@http://mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list


RUN apt-get update && apt-get install -y \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-cache-dir -r requirements.txt -i http://mirrors.tencentyun.com/pypi/simple --trusted-host mirrors.tencentyun.com

EXPOSE 5000
CMD ["python", "web_demo/app.py"]