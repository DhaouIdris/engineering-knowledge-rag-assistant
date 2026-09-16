from langchain.tools import Tool
import numexpr

def create_calculator_tool():
    def safe_eval(expr: str) -> str:
        try:
            return str(numexpr.evaluate(expr.strip()).item())
        except Exception as e:
            return f"Erreur de calcul: {e}"
    
    return Tool(
        name="Calculator",
        func=safe_eval,
        description="Utile pour effectuer des calculs mathématiques. Input: expression mathématique (ex: '2+2', '15*8/3')."
    )