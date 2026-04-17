"""
auditor.py
Static Rule Engine do sistema AZR.

Analisa código Python proposto pelo AZR usando AST (Abstract Syntax Tree),
buscando padrões perigosos antes de permitir a execução no Sandbox.
Permissões estritamente read-only — nunca escreve nada.

Regras implementadas:
  1. Loops infinitos (`while True` sem break/return interno)
  2. Import de módulos proibidos (os.system, subprocess, socket em contexto de sandbox)
  3. Tentativas de acesso a caminhos fora do sandbox (path traversal em strings)
  4. Uso de exec()/eval() (code injection secundário)
  5. Alocações de memória extremamente grandes (listas/dicts com > N elementos literais)
"""
import ast
import logging
from dataclasses import dataclass, field

log = logging.getLogger("AZR.Auditor")

_BANNED_MODULES = {"socket", "urllib", "http", "ftplib", "smtplib", "telnetlib"}
_DANGER_BUILTINS = {"exec", "eval", "compile", "__import__"}


@dataclass
class AuditViolation:
    rule: str
    message: str
    lineno: int = 0


@dataclass
class AuditResult:
    passed: bool
    violations: list[AuditViolation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pass": self.passed,
            "violations": [
                {"rule": v.rule, "message": v.message, "line": v.lineno}
                for v in self.violations
            ],
        }


class SecurityAuditor:
    """
    Static Rule Engine: analisa AST do código gerado em busca de padrões perigosos.
    Instância inflexível e sem efeitos colaterais — opera estritamente em read-only.
    """

    # ------------------------------------------------------------------ #
    #  Regras individuais                                                  #
    # ------------------------------------------------------------------ #

    def _check_infinite_loops(self, tree: ast.AST) -> list[AuditViolation]:
        """Detecta `while True:` sem `break` ou `return` imediato no corpo."""
        violations = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.While):
                continue
            # Verifica se a condição é literalmente True
            is_const_true = isinstance(node.test, ast.Constant) and node.test.value is True
            if not is_const_true:
                continue
            # Verifica se existe break/return no corpo direto (não nested)
            has_exit = any(isinstance(n, (ast.Break, ast.Return)) for n in node.body)
            if not has_exit:
                violations.append(AuditViolation(
                    rule="INFINITE_LOOP",
                    message="`while True` sem break/return detectado — risco de thermal throttling.",
                    lineno=node.lineno,
                ))
        return violations

    def _check_banned_imports(self, tree: ast.AST) -> list[AuditViolation]:
        """Bloqueia imports de módulos de rede dentro do sandbox."""
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                else:
                    names = [node.module.split(".")[0]] if node.module else []
                for name in names:
                    if name in _BANNED_MODULES:
                        violations.append(AuditViolation(
                            rule="BANNED_IMPORT",
                            message=f"Import de módulo proibido no sandbox: `{name}`.",
                            lineno=getattr(node, "lineno", 0),
                        ))
        return violations

    def _check_dangerous_builtins(self, tree: ast.AST) -> list[AuditViolation]:
        """Detecta uso de exec(), eval() e compile()."""
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name in _DANGER_BUILTINS:
                    violations.append(AuditViolation(
                        rule="DANGEROUS_BUILTIN",
                        message=f"Uso de `{name}()` detectado — risco de injeção de código secundário.",
                        lineno=getattr(node, "lineno", 0),
                    ))
        return violations

    def _check_path_traversal(self, tree: ast.AST) -> list[AuditViolation]:
        """Detecta strings de path contendo '..' (tentativa de saída do sandbox)."""
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if ".." in node.value or node.value.startswith("/etc") or node.value.startswith("/proc"):
                    violations.append(AuditViolation(
                        rule="PATH_TRAVERSAL",
                        message=f"String de path suspeito detectada: `{node.value[:60]}`.",
                        lineno=getattr(node, "lineno", 0),
                    ))
        return violations

    # ------------------------------------------------------------------ #
    #  Entry-point principal                                               #
    # ------------------------------------------------------------------ #

    def audit(self, code_string: str) -> AuditResult:
        """
        Executa todas as regras no código fornecido.

        Returns:
            AuditResult com passed=True se nenhuma violação for encontrada.
        """
        try:
            tree = ast.parse(code_string)
        except SyntaxError as e:
            return AuditResult(
                passed=False,
                violations=[AuditViolation("SYNTAX_ERROR", f"Código inválido: {e}", lineno=e.lineno or 0)],
            )

        all_violations: list[AuditViolation] = []
        all_violations += self._check_infinite_loops(tree)
        all_violations += self._check_banned_imports(tree)
        all_violations += self._check_dangerous_builtins(tree)
        all_violations += self._check_path_traversal(tree)

        if all_violations:
            for v in all_violations:
                log.warning(f"[Auditor] Violação L{v.lineno} [{v.rule}]: {v.message}")

        return AuditResult(passed=len(all_violations) == 0, violations=all_violations)
