import os

class CodeEditor:
    """
    Ferramenta exclusiva para modificar arquivos de experimentos internamente.
    Estritamente contida em um diretório alvo isolado.
    """
    def __init__(self, sandbox_workspace: str):
        self.sandbox_workspace = sandbox_workspace
        if not os.path.exists(self.sandbox_workspace):
             os.makedirs(self.sandbox_workspace)

    def write_file(self, filepath: str, code: str) -> dict:
        # Jailbreak prevention:
        if ".." in filepath or filepath.startswith("/"):
            return {"error": "Path traversal attempt blocked."}

        full_path = os.path.join(self.sandbox_workspace, filepath)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(code)
        
        return {"success": True, "path": filepath}

    def read_file(self, filepath: str) -> dict:
        full_path = os.path.join(self.sandbox_workspace, filepath)
        if not os.path.exists(full_path):
             return {"error": "File not found"}
        
        with open(full_path, "r", encoding="utf-8") as f:
            return {"content": f.read()}
