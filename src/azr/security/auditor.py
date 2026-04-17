import ast

class SecurityAuditor:
    """
    Static Rule Engine: Avalia o código gerado em busca de padrões perigosos
    antes de permitir a ida para o sandbox. Permissões estritas de read-only.
    """
    def __init__(self):
        pass

    def check_for_infinite_loops(self, code_string: str) -> bool:
        """
        Gera uma AST e procura por loops WHILE True não guardados (simplificado).
        Retorna True caso detecte violação de segurança.
        """
        try:
            tree = ast.parse(code_string)
            for node in ast.walk(tree):
                if isinstance(node, ast.While):
                    if isinstance(node.test, ast.Constant) and node.test.value is True:
                        # Achou um while True
                        return True
            return False
        except SyntaxError:
            return True # Código inválido é barreira automática

    def audit(self, code_string: str) -> dict:
        if self.check_for_infinite_loops(code_string):
            return {"pass": False, "reason": "Possivel loop infinito detectado (while True)"}
        return {"pass": True}
