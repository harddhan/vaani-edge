FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
    numpy \
    scipy \
    PyYAML \
    websockets \
    pytest \
    pytest-asyncio

COPY server/ server/
COPY ml/features/ ml/features/
COPY ml/__init__.py ml/
COPY configs/ configs/

EXPOSE 8765

CMD ["python3", "-m", "server.main"]