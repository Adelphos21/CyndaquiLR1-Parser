# 1. Imagen Base
# Usamos una imagen oficial de Python ligera. 
# Asegúrate de que la versión (ej. 3.11) sea compatible con la tuya.
FROM python:3.11-slim

# 2. Establecer el Directorio de Trabajo
# Creamos un directorio /app dentro del contenedor para nuestro código.
WORKDIR /app

# 3. Instalar Dependencias
# Copiamos solo el archivo de requisitos primero.
# Esto aprovecha la caché de Docker: si no cambias las dependencias,
# no se volverán a instalar cada vez que cambies tu código.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar el Código de la Aplicación
# Copiamos el resto de los archivos (main.py, analyzer.py)
COPY . .

# 5. Exponer el Puerto
# Le decimos a Docker que el contenedor escuchará en el puerto 8000.
EXPOSE 8000

# 6. Comando de Ejecución
# El comando para iniciar la API cuando se inicie el contenedor.
# Usamos --host 0.0.0.0 para que sea accesible desde fuera del contenedor.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]