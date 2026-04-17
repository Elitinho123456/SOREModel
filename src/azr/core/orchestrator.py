class Orchestrator:
    """
    Governador de Computação. Restringe a fatia de recursos entre o sandbox e a produção.
    Aplica cooldowns em casos de falha.
    """
    def __init__(self):
        self.cooldown_list = {}
        self.hardware_budget = {
            "sandbox_percentage": 0.20,
            "production_percentage": 0.80
        }

    def can_run_experiment(self, component_name: str) -> bool:
        """Checa prioridade de ociosidade e bloqueios."""
        import time
        if component_name in self.cooldown_list:
            if time.time() < self.cooldown_list[component_name]:
                return False
            else:
                del self.cooldown_list[component_name]
        return True
        
    def add_cooldown(self, component_name: str, duration_sec: int):
        import time
        self.cooldown_list[component_name] = time.time() + duration_sec
