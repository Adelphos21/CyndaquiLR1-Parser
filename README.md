# LR(1) Parser API

Este proyecto expone un analizador sintáctico LR(1) como una API web usando FastAPI.

Permite enviar una gramática y una cadena de entrada para recibir un análisis completo, incluyendo:

* Tablas de análisis (ACTION/GOTO)
* Traza paso a paso del parser
* Árbol de Sintaxis Abstracta (AST)

---

## ⚙️ Ejecución

### 🧩 Requisitos

* Python 3.8+
* pip

---

### 🚀 Opcion 1: Ejecutar localmente

1. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Ejecuta el servidor:

   ```bash
   uvicorn main:app --reload
   ```

   Luego abre [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### 🐳 Opcion 2: Usar Docker

1. Construye la imagen:

   ```bash
   docker build -t parser-api .
   ```

2. Corre el contenedor:

   ```bash
   docker run -d -p 8000:8000 --name lr1-server parser-api
   ```

El servidor estará disponible en `http://localhost:8000`.

---

## 🧪 Endpoint `/analyze`

**URL:** `http://localhost:8000/analyze`
**Método:** `POST`

### 🔹 Ejemplo de petición JSON

```json
{
  "grammar": "S -> C C\nC -> c C\nC -> d",
  "input_string": "c d d"
}
```

### 🔹 Ejemplo de petición con cURL

```bash
curl -X 'POST' \
  'http://localhost:8000/analyze' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"grammar": "S -> C C\nC -> c C\nC -> d", "input_string": "c d d"}'
```

### 🔹 Probar en el navegador

Abre [http://localhost:8000/docs](http://localhost:8000/docs) para usar la interfaz interactiva de FastAPI.
