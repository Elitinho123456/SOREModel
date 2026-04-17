import json

class ToolCaller:
    """
    Abstração e Roteador das Tools da IA.
    Exige que cada chamada de Tool para Mutação ou Execução defina o ROI
    (Return on Investment) da alteração no Hardware.
    """
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.registered_tools = {}

    def register_tool(self, name: str, func):
        self.registered_tools[name] = func

    def invoke(self, payload: str) -> dict:
        """
        Payload esperado: JSON com {
            "tool_name": "...", 
            "args": {...}, 
            "roi_estimate": {"gain_type": "...", "estimated_value": "..."}
        }
        """
        try:
            data = json.loads(payload)
            tool_name = data.get("tool_name")
            roi = data.get("roi_estimate")

            if tool_name not in self.registered_tools:
                return {"error": "Tool não registrada"}

            # Orquestrador valida se o momento de hardware permite
            if not self.orchestrator.can_run_experiment(tool_name):
                return {"error": "Bloqueio do Governador. Hardware ocupado ou Tool em cooldown."}

            if tool_name == "run_sandbox_experiment" and not roi:
                return {"error": "Métrica de ROI é estritamente necessária para autorizar o Sandbox"}

            # Executa
            func = self.registered_tools[tool_name]
            result = func(**data.get("args", {}))
            return result
        except Exception as e:
            return {"error": str(e)}
