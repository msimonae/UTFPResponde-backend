FROM python:3.11-slim

WORKDIR /app

# 1. Instalar dependências de sistema
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# 2. Copiar e instalar pacotes Python e o gdown
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gdown

# 3. DOWNLOAD DO CÉREBRO DIGITAL (VETORIAL) VIA GOOGLE DRIVE
# O ID extraído do seu link: 1DXG70lEMgmSOLvTZ8_2895QLKP2Y6AWj
# O nome do arquivo deve ser exatamente 'sklearn_index_ppgi.parquet' para o main.py funcionar
RUN gdown "https://drive.google.com/uc?id=1DXG70lEMgmSOLvTZ8_2895QLKP2Y6AWj" -O sklearn_index_ppgi.parquet

# 4. Copiar o restante do código da API
COPY . .

# 5. Expor a porta padrão do Cloud Run
EXPOSE 8080

# 6. Iniciar o servidor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
