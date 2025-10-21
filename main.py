# main.py

import collections
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from analyzer import LR1Analyzer, ASTNode
except ImportError:
    print("ERROR: No se pudo encontrar el archivo 'analyzer.py'.")
    exit()

#--- Definición del Servidor ---
app = FastAPI(
    title="LR(1) Parser API",
    description="Una API para generar tablas de análisis LR(1) y parsear cadenas."
)


# --- Modelo de Datos de Entrada ---
class AnalysisRequest(BaseModel):
    """Define la forma esperada del JSON en el body de la petición.
    {
        "grammar": "S -> C C\\nC -> c C\\nC -> d",
        "input_string": "c d d"
    }
    Utiliza saltos de línea (\\n) para separar las producciones.
    """
    grammar: str
    input_string: str

    # Ejemplo de cómo se vería el JSON
    class Config:
        schema_extra = {
            "example": {
                "grammar": "S -> C C\nC -> c C\nC -> d",
                "input_string": "c d d"
            }
        }


# --- Definición de Endpoints ---

@app.post("/analyze")
async def run_analysis(request_data: AnalysisRequest):
    """
    Recibe la gramática y la cadena,
    y devuelve el resultado completo del análisis.
    """
    
    try:
        # Ejecutar el analizador LR(1)
        analyzer = LR1Analyzer(request_data.grammar)
        results = analyzer.analyze(request_data.input_string)
        
        # Limpiar los resultados para que sean serializables en JSON
        
        if results.get("ast"):
            # Convertimos el AST a su representación en string
            results["ast"] = str(results["ast"])
        else:
            results["ast"] = "No se generó AST."

        # Convertimos los defaultdicts a dicts normales
        results["action_table"] = dict(results["action_table"])
        results["goto_table"] = dict(results["goto_table"])

        return results

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error de Gramática: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error inesperado en el análisis: {e}"
        )

# uvicorn main:app --reload