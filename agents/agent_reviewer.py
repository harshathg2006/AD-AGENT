import os
import re
import subprocess
import sys
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from utils.gemini_client import query_gemini_quota_safe, query_gemini_with_retry
from langchain_core.prompts import PromptTemplate

# ------------------------ Updated Test Prompt ------------------------
test_prompt = PromptTemplate.from_template("""
You will receive a Python script for {package_name} that trains an anomaly-detection model.

--- BEGIN CODE ---
{code}
--- END CODE ---

TASK:
1. Replace all data-loading operations with synthetic data creation code.
2. Maintain all original **imports, logic, model parameters, and variable names** exactly.
3. **Synthetic Data Rules:**
    * **PyOD models:**
        - Import: `from pyod.utils.data import generate_data`
        - Use this exact line to create synthetic data:
            `X_train_synth, _, y_train_synth, _ = generate_data(
                n_train=2000, n_test=0, n_features=187, contamination=0.1, random_state=42)`
        - Replace original dataset loading with:
            `X_train, y_train, X_test, y_test = X_train_synth, y_train_synth, None, None`
        - **Ensure all model parameters match synthetic data dimensions, especially `n_features=X_train.shape[1]`.**
    * **PyGOD/Darts/TSLib:** follow original instructions for synthetic replacement.

4. Ensure the code runs standalone:
    - Compute metrics like AUROC/AUPRC if possible; else print placeholder metrics.
    - Do not remove or change variable names or model parameters unnecessarily.

5. Output **only runnable Python code**. No markdown, no explanation, no extra text.
""")

# ------------------------ Agent Reviewer ------------------------
class AgentReviewer:
    """Verifies generated code using Gemini and produces final executable code."""

    MAX_RETRIES = 2

    def __init__(self):
        pass

    # ------------------------ Utility Functions ------------------------
    @staticmethod
    def print_python_code(code_str):
        """Pretty print Python code with syntax highlighting."""
        print(highlight(code_str, PythonLexer(), TerminalFormatter()))

    @staticmethod
    def _clean_markdown(txt: str) -> str:
        """Extract Python code from Gemini responses (strip Markdown fences)."""
        match = re.search(r"```(?:python)?\n(.*?)```", txt, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        txt = re.sub(r"```(python)?", "", txt)
        txt = re.sub(r"```", "", txt)
        cleaned_lines = []
        for line in txt.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.lower().startswith(("response:", "output:", "note:", "explanation:", "here is the fix")):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    # ------------------------ Core Function ------------------------
    def test_code(self, code: str, algorithm_name: str, package_name: str):
        """
        Test the generated code using synthetic data.
        If errors occur, automatically prompt Gemini (with CoT reasoning) to fix them iteratively.
        """
        cleaned_code = code
        folder = "generated_scripts"
        os.makedirs(folder, exist_ok=True)
        script_path = os.path.join(folder, f"{algorithm_name}_test.py")

        for attempt in range(1, self.MAX_RETRIES + 1):
            print(f"\n=== [Reviewer] Attempt {attempt} for {algorithm_name} ({package_name}) ===")

            # --- Step 1: Construct CoT-based synthetic test generation prompt ---
            test_prompt_cot = f"""
You are an expert Python ML developer testing anomaly detection algorithms.

TASK: Modify or extend the given script so it can run standalone with synthetic data
(you can use numpy or sklearn.datasets.make_classification).

--- ORIGINAL CODE ---
{cleaned_code}
--- END CODE ---

ALGORITHM: {algorithm_name}
PACKAGE: {package_name}

THINK STEP BY STEP:
1. Generate small synthetic data (e.g., 200 samples, 10 features).
2. Ensure the model trains and evaluates correctly.
3. Compute AUROC/AUPRC if possible, else print placeholder metrics.
4. Maintain the same variable names, structure, and imports.
5. **IMPORTANT:** For models like DeepSVDD, always pass `n_features=X_train.shape[1]` when initializing.
6. Return the **entire runnable Python script only** (no markdown, no explanation).
"""

            # --- Step 2: Query Gemini for synthetic testable variant ---
            response = query_gemini_with_retry(test_prompt_cot)
            cleaned_code = self._clean_markdown(response)

            print("\n[DEBUG] Gemini Synthetic Test Code:\n")
            self.print_python_code(cleaned_code)

            # --- Step 3: Save and execute test script ---
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(cleaned_code)

            res = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
            print("\n=== [Execution Output] ===")
            print(res.stdout)
            if res.stderr:
                print("[stderr]", res.stderr)

            # --- Step 4: Check execution success ---
            if res.returncode == 0:
                print(f"✅ {algorithm_name} test passed successfully.\n")
                return True, cleaned_code

            # --- Step 5: If failed, build CoT-based fix prompt ---
            print(f"❌ {algorithm_name} test failed. Sending full code + error to Gemini for fix.")

            fix_prompt_cot = f"""
You are an expert Python debugger.
The following Python script failed during testing.

--- BEGIN CODE ---
{cleaned_code}
--- END CODE ---

Error encountered:
{res.stderr}

THINK STEP BY STEP:
1. Identify the exact cause of failure (imports, parameters, data handling, etc.).
2. Infer the correct fix from context.
3. Preserve logic, structure, and variable names.
4. Ensure it runs without any runtime or syntax errors.
5. **IMPORTANT:** For models like DeepSVDD, always pass `n_features=X_train.shape[1]` when initializing.
6. Output only the corrected runnable Python code (no markdown or explanation).
"""
            raw_fix = query_gemini_with_retry(fix_prompt_cot)
            cleaned_code = self._clean_markdown(raw_fix)

        print(f"❌ {algorithm_name} could not be fixed after {self.MAX_RETRIES} attempts.\n")
        return False, cleaned_code




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