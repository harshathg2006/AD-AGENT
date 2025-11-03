# agents/agent_code_generator.py
import os
import re
import sys
import ast
import subprocess
import importlib
import inspect
from datetime import datetime, timedelta
from typing import Optional

import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter

from entity.code_quality import CodeQuality
from config.config import Config
from utils.gemini_client import query_gemini_with_retry  # retry-aware Gemini call

# Configure Gemini client
genai.configure(api_key=Config.GEMINI_API_KEY)

# ---- Helpers --------------------------------------------------------------

def print_python_code(code_str: str) -> None:
    """Pretty print Python code with syntax highlighting in the terminal."""
    try:
        print(highlight(code_str, PythonLexer(), TerminalFormatter()))
    except Exception:
        print(code_str)


def extract_python_code(response_text: str) -> str:
    """
    Extract and clean Python code from Gemini's response.
    Adds AUROC/AUPRC metrics and auto-saves model if missing.
    """
    code_match = re.search(r"```(?:python)?\n(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    code = code_match.group(1) if code_match else response_text

    cleaned_lines = []
    for line in code.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith(("response:", "output:", "note:", "explanation:", "[debug]")):
            continue
        cleaned_lines.append(line)
    cleaned_code = "\n".join(cleaned_lines).strip()

    required_imports = [
        "import numpy as np",
        "from sklearn.metrics import roc_auc_score, average_precision_score",
        "import joblib"
    ]
    # If the code contains no imports, prepend required imports
    if not any(line.startswith("import ") or line.startswith("from ") for line in cleaned_code.splitlines()):
        cleaned_code = "\n".join(required_imports) + "\n\n" + cleaned_code

    # Add AUROC/AUPRC metrics if missing
    if "auroc_score" not in cleaned_code or "auprc_score" not in cleaned_code:
        metric_code = (
            "\n# Added missing metrics\n"
            "try:\n"
            "    y_test_scores = model.decision_function(X_test)\n"
            "except Exception:\n"
            "    y_test_scores = model.score_samples(X_test) if hasattr(model, 'score_samples') else np.zeros(len(X_test))\n"
            "auroc_score = roc_auc_score(y_test, y_test_scores)\n"
            "auprc_score = average_precision_score(y_test, y_test_scores)\n"
            'print(f"AUROC: {auroc_score}")\n'
            'print(f"AUPRC: {auprc_score}")\n'
        )
        cleaned_code += "\n\n" + metric_code

    # Auto-save model if model.fit present but joblib.dump missing
    if "model.fit" in cleaned_code and "joblib.dump" not in cleaned_code:
        save_snippet = (
            "\n# Auto-save trained model\n"
            "try:\n"
            "    joblib.dump(model, f'trained_model.pkl')\n"
            "    print('Model saved successfully!')\n"
            "except Exception as e:\n"
            "    print('Warning: failed to save model:', e)\n"
        )
        cleaned_code = re.sub(r"(model\.fit\([^\)]*\))", r"\1\n" + save_snippet, cleaned_code, flags=re.DOTALL)

    # Ensure script starts with valid Python token
    if not cleaned_code.startswith(("import ", "from ", "def ", "class ", "#")):
        cleaned_code = "# Auto-fixed script\n" + cleaned_code

    return cleaned_code


# ---- Prompt templates -----------------------------------------------------

# PyOD
template_pyod_labeled = PromptTemplate.from_template("""
You are an expert developer. Produce executable Python only (no explanation).

Context: PyOD anomaly detection algorithm `{algorithm}`.
TRAIN CSV path: {data_path_train}
TEST CSV path: {data_path_test}
Doc excerpt:
{algorithm_doc}
                                                     
Environment:
- Python 3.11
- Windows
- pyod==2.0.5                                                   

Requirements:
- All code must be compatible with the environment above.
- Import sys, os and add sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
- from data_loader.data_loader import DataLoader
- Instantiate DataLoader for train/test, call load_data(split_data=False) to obtain X_train, y_train, X_test, y_test
- Initialize `model = {algorithm}(<only use parameters from {parameters} that match the class signature>)`
- Fit model on X_train
- Compute train/test scores and compute AUROC/AUPRC and print them
- Print only runnable Python code.
""")

template_pyod_unlabeled = PromptTemplate.from_template("""
You are an expert developer. Produce executable Python only (no explanation).

Context: PyOD algorithm `{algorithm}` on unlabeled data.
TRAIN CSV path: {data_path_train}
Doc excerpt:
{algorithm_doc}

Environment:
- Python 3.11
- Windows
- pyod==2.0.5                                                   

Requirements:
- All code must be compatible with the environment above.
- Load dataset using DataLoader, assign X_train
- Initialize `{algorithm}` and fit on X_train
- Compute scores on X_train and print placeholders AUROC/AUPRC (if labels absent)
- Output runnable Python only.
""")

# PyGOD
template_pygod_labeled = PromptTemplate.from_template("""
You are an expert developer. Produce executable Python only.

Context: PyGOD anomaly detection algorithm `{algorithm}`.
TRAIN CSV path: {data_path_train}
TEST CSV path: {data_path_test}
Doc excerpt:
{algorithm_doc}

Environment:
- Python 3.11
- Windows
- pygod

Requirements:
- Use DataLoader to load train/test datasets
- Initialize `{algorithm}` with matching parameters
- Fit model, compute train/test scores, print AUROC/AUPRC
- Output only executable Python code
""")

template_pygod_unlabeled = PromptTemplate.from_template("""
You are an expert developer. Produce executable Python only.

Context: PyGOD algorithm `{algorithm}` on unlabeled data.
TRAIN CSV path: {data_path_train}
Doc excerpt:
{algorithm_doc}

Requirements:
- Use DataLoader to load train data
- Initialize `{algorithm}`, fit on X_train
- Compute scores and print placeholders AUROC/AUPRC
""")

# TSLib
template_tslib_labeled = PromptTemplate.from_template("""
You are an expert developer. Produce executable Python only.

Context: TSLib forecasting algorithm `{algorithm}`.
TRAIN CSV path: {data_path_train}
TEST CSV path: {data_path_test}
Doc excerpt:
{algorithm_doc}

Requirements:
- Use DataLoader to load datasets
- Initialize `{algorithm}` with matching parameters
- Fit model and compute predictions
- Print results
""")

# Darts
template_darts_labeled = PromptTemplate.from_template("""
You are an expert developer. Produce executable Python only.

Context: Darts forecasting model `{algorithm}`.
TRAIN CSV path: {data_path_train}
TEST CSV path: {data_path_test}
Doc excerpt:
{algorithm_doc}

Requirements:
- Load datasets with DataLoader
- Initialize `{algorithm}` with parameters
- Fit model on train, predict on test
- Compute metrics (AUROC/AUPRC if labels present)
- Output only executable Python
""")

template_darts_unlabeled = PromptTemplate.from_template("""
You are an expert developer. Produce executable Python only.

Context: Darts model `{algorithm}` on unlabeled data.
TRAIN CSV path: {data_path_train}
Doc excerpt:
{algorithm_doc}

Requirements:
- Load dataset with DataLoader
- Fit `{algorithm}` on train only
- Compute scores, print placeholders AUROC/AUPRC
""")

# Fix template (your advanced CoT-style prompt)
template_fix = PromptTemplate.from_template("""
You are an expert Python ML developer specializing in anomaly detection and time-series modeling.
You will analyze and fix a Python script that failed to run.

---

### 🧩 Step 1: Deep Reasoning
Think step-by-step about:
- What this code tries to do
- Why the given error occurred (parameter mismatch, version issue, etc.)
- How to fix it using valid library APIs
- How to preserve existing logic and data flow

Do NOT output your thoughts — just use them internally to reason.

---

### 🧰 Step 2: Apply the fix
Now repair the code so that:
- It runs successfully without syntax or runtime errors.
- It uses valid parameters and method names from the latest API (per doc excerpt below).
- It keeps model, training, and evaluation structure unchanged.
- It prints or saves results properly.

---

### ⚙️ Step 3: Constraints
- Use Python 3.11 syntax only.
- Ensure imports are valid and minimal.
- No pseudo-code, no comments like "TODO".
- Return ONLY runnable Python code (no markdown fences, no explanation).

---

### Error Message
{error_message}

### Algorithm Documentation (excerpt)
{algorithm_doc}

### Original Code
{code}
""")

# ---- AgentCodeGenerator class ---------------------------------------------

class AgentCodeGenerator:
    """Responsible for generating and revising runnable model scripts using Gemini."""

    def __init__(self):
        pass

    def generate_code(
        self,
        algorithm: str,
        data_path_train: str,
        data_path_test: Optional[str],
        algorithm_doc: str,
        input_parameters: dict,
        package_name: str
    ) -> str:
        """Generate runnable Python code for the specified algorithm and dataset(s)."""

        # ---- Step 0: Dynamic parameter filtering ----
        def filter_valid_params(pkg: str, alg: str, params: dict) -> dict:
            filtered = {}
            try:
                if pkg == "pyod":
                    mod = importlib.import_module(f"pyod.models.{alg.lower()}")
                    cls = getattr(mod, alg)
                    sig = inspect.signature(cls.__init__)
                    filtered = {k: v for k, v in (params or {}).items() if k in sig.parameters}
                elif pkg == "darts":
                    mod = importlib.import_module(f"darts.models.forecasting.{alg.lower()}")
                    cls = getattr(mod, alg)
                    sig = inspect.signature(cls.__init__)
                    filtered = {k: v for k, v in (params or {}).items() if k in sig.parameters}
                else:
                    filtered = params or {}
            except Exception as e:
                print(f"[Warning] Could not filter params for {alg}: {e}")
                filtered = params or {}
            return filtered

        filtered_params = filter_valid_params(package_name, algorithm, input_parameters)

        # ---- Step 0.5: Inject n_features for DeepSVDD ----
        if package_name == "pyod" and algorithm.lower() == "deepsvdd" and 'n_features' not in filtered_params:
            try:
                # Attempt to use project's DataLoader if available
                from data_loader.data_loader import DataLoader
                dl = DataLoader(data_path_train)
                df_train = dl.load_data(split_data=False)
                # if load_data returns a tuple (X, y, ...) or np.ndarray, handle safely
                if hasattr(df_train, "columns"):
                    df_tmp = df_train
                elif isinstance(df_train, tuple):
                    # sometimes load_data returns (X, y) arrays; convert to shape
                    # can't deduce column names from numpy arrays, so infer n_features directly
                    if hasattr(df_train[0], "shape"):
                        n_features = df_train[0].shape[1]
                        filtered_params['n_features'] = n_features
                        print(f"[INFO] Injected n_features={n_features} for DeepSVDD (from tuple load_data)")
                else:
                    # if it's a pandas-like object convertible to DataFrame
                    try:
                        import pandas as pd
                        df_tmp = pd.DataFrame(df_train)
                    except Exception:
                        df_tmp = None

                if 'n_features' not in filtered_params and df_tmp is not None:
                    for c in ['Time', 'Class']:
                        if c in df_tmp.columns:
                            df_tmp = df_tmp.drop(columns=c)
                    n_features = df_tmp.shape[1]
                    filtered_params['n_features'] = n_features
                    print(f"[INFO] Injected n_features={n_features} for DeepSVDD")
            except Exception as e:
                print(f"[Warning] Failed to auto-inject n_features: {e}")

        # ---- Step 1: Select prompt template ----
        if package_name == "pyod":
            tpl = template_pyod_labeled if data_path_test else template_pyod_unlabeled
        elif package_name == "pygod":
            tpl = template_pygod_labeled if data_path_test else template_pygod_unlabeled
        elif package_name == "tslib":
            tpl = template_tslib_labeled
        else:  # darts
            tpl = template_darts_labeled if data_path_test else template_darts_unlabeled

        # ---- Step 2: Compose prompt ----
        prompt_vars = {
            "algorithm": algorithm,
            "data_path_train": data_path_train,
            "data_path_test": data_path_test or "",
            "algorithm_doc": algorithm_doc or "",
            "parameters": filtered_params
        }
        prompt = tpl.format(**prompt_vars)

        # ---- Step 3: Debug prompt ----
        print("\n[DEBUG] GEMINI PROMPT (truncated 2k chars):\n")
        print(prompt[:2000] + ("..." if len(prompt) > 2000 else ""))

        # ---- Step 4: Query Gemini ----
        raw_text = query_gemini_with_retry(prompt)

        # ---- Step 5: Debug raw output ----
        print("\n[DEBUG] GEMINI RAW TEXT (codegen):\n")
        print_python_code(raw_text)

        # ---- Step 6: Extract and clean Python code ----
        final_code = extract_python_code(raw_text)

        # ---- Step 7: Debug cleaned code ----
        print("\n[DEBUG] Cleaned Gemini code:\n")
        print_python_code(final_code)

        return final_code

    def revise_code(self, code_quality: CodeQuality, algorithm_doc: str) -> str:
        """
        Request Gemini to fix a failing script using CoT-style prompt and retry logic.
        Returns cleaned code (best-effort). Increments code_quality.review_count.
        """

        # Defensive extraction of algorithm name for logging
        try:
            alg_name = getattr(code_quality, "algorithm_name", None) or getattr(code_quality, "algorithm", None)
            if isinstance(alg_name, (list, tuple)):
                alg_name = alg_name[0] if alg_name else "UnknownAlg"
        except Exception:
            alg_name = "UnknownAlg"

        # Prepare the CoT-based fix prompt using the template_fix PromptTemplate
        # We'll attempt multiple times if Gemini returns code that still errors.
        MAX_FIX_ATTEMPTS = 3
        last_cleaned = code_quality.code or ""
        last_raw = ""
        for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
            print(f"\n=== [REVISION ATTEMPT {attempt}] for {alg_name} ===")
            prompt = template_fix.format(
                code=last_cleaned,
                error_message=code_quality.error_message or "",
                algorithm_doc=algorithm_doc or ""
            )

            print("\n[DEBUG] Sending fix prompt to Gemini (truncated 2k chars):\n")
            print(prompt[:2000] + ("..." if len(prompt) > 2000 else ""))

            try:
                raw_fix = query_gemini_with_retry(prompt)
                last_raw = raw_fix
                print("\n[DEBUG] GEMINI RAW TEXT (fix):\n")
                print_python_code(raw_fix)
            except Exception as e:
                print(f"[Warning] Gemini fix call failed: {e}")
                # If query fails, break and return current best
                break

            # Extract/clean code from Gemini response
            cleaned = extract_python_code(raw_fix)

            # Quick sanity checks: ensure it's python-ish
            if not cleaned.strip():
                print("[Warning] Gemini returned empty or non-code response. Retrying...")
                last_cleaned = last_cleaned  # keep previous
                continue

            # Save cleaned temporarily and attempt to run basic syntax check
            try:
                compile(cleaned, "<string>", "exec")
            except SyntaxError as se:
                print(f"[Warning] Fix produced syntax error: {se}. Trying next attempt.")
                last_cleaned = cleaned
                continue

            # Optionally run a lightweight lint or run tests — here we trust cleaned code
            # Increase review count and return the cleaned result
            code_quality.review_count = getattr(code_quality, "review_count", 0) + 1
            print(f"[INFO] Revision success (attempt {attempt}). Returning fixed code.")
            print("\n[DEBUG] Cleaned fixed code (first 1k chars):\n")
            print_python_code(cleaned[:1000] + ("..." if len(cleaned) > 1000 else ""))
            return cleaned

        # If we exhausted attempts, increment review_count and return last cleaned attempt
        code_quality.review_count = getattr(code_quality, "review_count", 0) + 1
        print(f"[Error] Could not produce a fully fixed script after {MAX_FIX_ATTEMPTS} attempts.")
        # As a last-ditch, return the last_raw extraction if available
        fallback = extract_python_code(last_raw) if last_raw else last_cleaned
        return fallback

    @staticmethod
    def _clean(code: str) -> str:
        code = re.sub(r"```(?:python)?", "", code, flags=re.IGNORECASE)
        return re.sub(r"```", "", code).strip()

    @staticmethod
    def _extract_init_params_dict(response_text: str) -> dict:
        match = re.search(r"```python\s*({.*?})\s*```", response_text, re.DOTALL)
        if not match:
            return {}
        dict_str = match.group(1)
        try:
            return ast.literal_eval(dict_str)
        except Exception:
            return {}

# Quick smoke test
if __name__ == "__main__":
    gen = AgentCodeGenerator()
    print("AgentCodeGenerator ready with DeepSVDD n_features support.")
