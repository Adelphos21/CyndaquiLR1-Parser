# 1. Imagen base
FROM python:3.11-slim

# 2. Directorio de trabajo
WORKDIR /app

# 3. Dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar código
COPY . .

# 5. Exponer puerto (solo informativo)
EXPOSE 8000

# 6. Comando
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
