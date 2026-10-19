FROM python:3.11-slim

WORKDIR /app

# Instala dependências essenciais do sistema
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copia e instala as dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código fonte para dentro do container
COPY . .

EXPOSE 8080

# Comando robusto apontando explicitamente para main:app e respeitando a porta dinâmica do Cloud Run
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
