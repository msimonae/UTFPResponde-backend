# 1. Usa um sistema Linux leve com Python
FROM python:3.11-slim

# 2. Define a pasta de trabalho
WORKDIR /app

# 3. Instala ferramentas de sistema necessárias
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# 4. Copia as dependências e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gdown

# 5. O SEGREDO: Baixa o banco vetorial do Google Drive direto para o servidor
RUN gdown "1DXG70lEMgmSOLvTZ8_2895QLKP2Y6AWj" -O sklearn_index_ppgi_v11.parquet

# 6. Copia o resto do código (main.py, etc)
COPY . .

# 7. Inicia a API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
