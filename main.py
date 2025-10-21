import collections
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    # Esta importación ahora funcionará porque analyzer.py existe
    from analyzer import LR1Analyzer, ASTNode, serialize_ast_to_graph
except ImportError:
    print("ERROR: Asegúrate de que el archivo 'analyzer.py' está en la misma carpeta.")
    exit()

# --- Definición del Servidor ---
app = FastAPI(
    title="LR(1) Parser API",
    description="Una API para generar tablas de análisis LR(1) y parsear cadenas."
)


# --- Modelo de Datos de Entrada ---
class AnalysisRequest(BaseModel):
    """Define la forma esperada del JSON en el body de la petición."""
    grammar: str
    input_string: str

    class Config:
        schema_extra = {
            "example": {
                "grammar": "S -> C C\nC -> c C | d",
                "input_string": "c d c d"
            }
        }


# --- Definición de Endpoints ---

@app.post("/analyze")
async def run_analysis(request_data: AnalysisRequest):
    """
    Recibe la gramática y la cadena, y devuelve el resultado completo del análisis.
    """

    try:
        # 1. Ejecutar el analizador LR(1)
        analyzer = LR1Analyzer(request_data.grammar)
        results = analyzer.analyze(request_data.input_string)

        # 2. Procesar el AST para que sea JSON serializable
        ast_object = results.get("ast")

        if ast_object:
            results["ast_string"] = str(ast_object)
            results["ast_graph"] = serialize_ast_to_graph(ast_object)
            del results["ast"]  # El objeto original no es serializable
        else:
            results["ast_string"] = "No se generó AST."
            results["ast_graph"] = {"nodes": [], "edges": []}
            if "ast" in results:
                del results["ast"]

        # 3. Convertir defaultdicts a dicts normales para la respuesta JSON
        results["action_table"] = dict(results["action_table"])
        results["goto_table"] = dict(results["goto_table"])

        return results

    except ValueError as e:
        # Captura errores de gramática (ej. conflictos S/R)
        raise HTTPException(
            status_code=400,  # Bad Request
            detail=f"Error en la Gramática o el Análisis: {e}"
        )
    except Exception as e:
        # Captura cualquier otro error inesperado
        raise HTTPException(
            status_code=500,  # Internal Server Error
            detail=f"Error inesperado en el servidor: {e}"
        )
