import os
import re
import subprocess
import sys
import ast
import importlib.util
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter

from entity.code_quality import CodeQuality
from utils.gemini_client import query_gemini_quota_safe  # your Gemini wrapper

def print_python_code(code_str):
    """Pretty-print Python code in the terminal."""
    print(highlight(code_str, PythonLexer(), TerminalFormatter()))

class AgentEvaluator:
    """
    Executes code with real data, optionally parses AUROC/AUPRC (skip for unsupervised),
    auto-installs missing libs, retries via Gemini if execution fails.
    """

    MAX_RETRIES = 2

    def execute_code(self, code: str, algorithm_name: str, unsupervised: bool = False) -> CodeQuality:
        """Execute code and automatically fix errors with Gemini if needed."""
        cleaned_code = self._clean_markdown(code)

        for attempt in range(1, self.MAX_RETRIES + 1):
            print(f"\n=== [Evaluator] Attempt {attempt} for {algorithm_name} ===")
            self._ensure_dependencies(cleaned_code)

            folder = "./generated_scripts"
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"{algorithm_name}.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(cleaned_code)

            # Run the script
            res = subprocess.run([sys.executable, path], capture_output=True, text=True)
            print("\n=== Execution Output ===\n", res.stdout, res.stderr)

            if res.returncode == 0:
                # Success: parse metrics only if not unsupervised
                auroc, auprc = -1, -1
                if not unsupervised:
                    auroc  = self._find_float(r"AUROC:\s*([\d.]+)", res.stdout)
                    auprc  = self._find_float(r"AUPRC:\s*([\d.]+)", res.stdout)

                errors = self._parse_errors(res.stdout)

                return CodeQuality(
                    code=cleaned_code,
                    algorithm=algorithm_name,
                    parameters={},
                    std_output=res.stdout,
                    error_message="",
                    auroc=auroc,
                    auprc=auprc,
                    error_points=errors,
                    review_count=0
                )
            else:
                # Execution failed, send code + error to Gemini
                print(f"[ERROR] Attempt {attempt} failed, sending to Gemini for fix.")
                prompt = f"""
You are a Python expert. The following script failed with an error:

--- BEGIN CODE ---
{cleaned_code}
--- END CODE ---

Error encountered:
{res.stderr}

TASK:
1. Fix the error and return runnable Python code.
2. Keep variable names and logic unchanged.
3. Output only executable Python code (no markdown or explanation).
"""
                raw_fix = query_gemini_quota_safe(prompt)
                cleaned_code = self._clean_markdown(raw_fix)

        # If all retries fail, return last error
        return CodeQuality(
            code=cleaned_code,
            algorithm=algorithm_name,
            parameters={},
            std_output=res.stdout,
            error_message=res.stderr,
            auroc=-1,
            auprc=-1,
            error_points=[],
            review_count=0
        )


    # ---------- deps handling ----------
    def _ensure_dependencies(self, code_str: str) -> None:
        """Auto-install missing top-level packages."""
        modules = self._discover_imports(code_str)
        if not modules:
            return

        module_to_pip = {
            "numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "sklearn": "scikit-learn",
            "statsmodels": "statsmodels", "matplotlib": "matplotlib", "seaborn": "seaborn",
            "pyod": "pyod", "pygod": "pygod", "darts": "u8darts",
            "torch": "torch", "torchvision": "torchvision", "tensorflow": "tensorflow",
            "pytorch_lightning": "pytorch-lightning",
            "cv2": "opencv-python", "PIL": "Pillow", "skimage": "scikit-image",
            "xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost",
        }

        stdlib = set(getattr(sys, "stdlib_module_names", set())) or {
            "os","sys","time","math","random","datetime","re","json","subprocess",
            "itertools","functools","collections","typing","pathlib","csv","ast"
        }

        to_install = [
            module_to_pip.get(m, m) for m in modules
            if m not in stdlib and importlib.util.find_spec(m) is None
        ]

        if to_install:
            print(f"[INFO] Installing missing packages: {to_install}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", *to_install])
        else:
            print("[INFO] All required third-party packages already available.")

    def _discover_imports(self, code_str: str) -> set:
        """Return set of top-level imports."""
        try:
            tree = ast.parse(code_str)
        except SyntaxError:
            return set(re.findall(r"^\s*(?:import|from)\s+([a-zA-Z0-9_]+)", code_str, re.MULTILINE))

        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update((alias.name.split(".")[0] for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    mods.add(node.module.split(".")[0])
        return mods

    # ---------- helpers ----------
    @staticmethod
    def _clean_markdown(txt: str) -> str:
        # Extract code from fenced block
        match = re.search(r"```(?:python)?\n(.*?)```", txt, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback: remove fences and explanations if no block is found
        txt = re.sub(r"```(python)?", "", txt)
        txt = re.sub(r"```", "", txt)

        # Remove common explanation lines
        cleaned_lines = []
        for line in txt.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.lower().startswith(("response:", "output:", "note:", "explanation:", "here is the fix")):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def _find_float(pattern: str, text: str, default: float = -1.0) -> float:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else default

    @staticmethod
    def _parse_errors(text: str):
        pts = []
        for line in text.splitlines():
            if "Failed prediction at point" in line:
                m = re.search(r"\[([^\]]+)] with true label ([\d.]+)", line)
                if m:
                    nums = [float(x.strip()) for x in m.group(1).split(",")]
                    pts.append({"point": nums, "true_label": float(m.group(2))})
        return pts
