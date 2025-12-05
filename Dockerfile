FROM python:3.11-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt -i http://mirrors.tencentyun.com/pypi/simple --trusted-host mirrors.tencentyun.com

EXPOSE 5000
CMD ["python", "web_demo/app.py"]