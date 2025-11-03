"""
AgentProcessor - Few-shot CoT Extraction with Automatic Quota Handling
-----------------------------------------------------------------------
Handles user dialogue, Few-shot CoT extraction, and retries Gemini API on quota errors.
"""

import os
import re
import json
import sys
from utils.gemini_client import query_gemini
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class AgentProcessor:
    # ---------- Few-shot Chain-of-Thought Examples ----------
    FEW_SHOT_COT_PROMPT = [
        {
            "role": "system",
            "content": (
                "You are an extraction assistant. "
                "Read the USER_INSTRUCTION and return four fields:\n"
                "• algorithm (array) • dataset_train • dataset_test • parameters (object)\n"
                "First show your reasoning, THEN output one line starting with FINAL: "
                "followed by the JSON dictionary. Do NOT invent names."
            ),
        },
        # Example 1
        {"role": "user", "content": "Run IForest on ./data/train.mat and ./data/test.mat with contamination=0.1"},
        {"role": "assistant", "content": (
            "Step 1 ▶ algorithm → IForest\n"
            "Step 2 ▶ train → ./data/train.mat ; test → ./data/test.mat\n"
            "Step 3 ▶ param → contamination=0.1\n"
            'FINAL: {"algorithm":["IForest"],"dataset_train":"./data/train.mat",'
            '"dataset_test":"./data/test.mat","parameters":{"contamination":0.1}}'
        )},
        # Example 2
        {"role": "user", "content": "Run DeepSVDD and IForest with contamination=0.2, max_iter=300 on ./train.csv for training and ./test.csv for evaluation"},
        {"role": "assistant", "content": (
            "Step 1 ▶ algorithms → DeepSVDD, IForest\n"
            "Step 2 ▶ train → ./train.csv ; test → ./test.csv\n"
            "Step 3 ▶ params → contamination=0.2, max_iter=300\n"
            'FINAL: {"algorithm":["DeepSVDD","IForest"],"dataset_train":"./train.csv",'
            '"dataset_test":"./test.csv","parameters":{"contamination":0.2,"max_iter":300}}'
        )},
        # Example 3: Algorithm only
        {"role": "user", "content": "Run LOF"},
        {"role": "assistant", "content": (
            "Step 1 ▶ algorithm → LOF\n"
            "Step 2 ▶ no datasets\n"
            "Step 3 ▶ no parameters\n"
            'FINAL: {"algorithm":["LOF"],"dataset_train":null,"dataset_test":null,"parameters":{}}'
        )},
        # Example 4: Run all algorithms
        {"role": "user", "content": "Run all algorithms on ./d1.txt and ./d2.txt"},
        {"role": "assistant", "content": (
            "Step 1 ▶ algorithm keyword → all\n"
            "Step 2 ▶ train → ./d1.txt ; test → ./d2.txt\n"
            "Step 3 ▶ no parameters\n"
            'FINAL: {"algorithm":["all"],"dataset_train":"./d1.txt",'
            '"dataset_test":"./d2.txt","parameters":{}}'
        )},
        # Placeholder for actual user input
        {"role": "user", "content": "USER_INSTRUCTION:\n<START>\n{user_input}\n<END>"},
    ]

    def __init__(self):
        # Conversation history
        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant helping users specify algorithm experiments. "
                    "Ensure they provide the algorithm, datasets (training & testing), "
                    "and optional parameters before finalizing the configuration."
                ),
            }
        ]
        # Extracted experiment config
        self.experiment_config = {
            "algorithm": [],
            "dataset_train": "",
            "dataset_test": "",
            "parameters": {},
        }

    def get_gemini_response(self, messages):
        """
        Converts few-shot messages into a text prompt and queries Gemini.
        Handles quota delays automatically via query_gemini().
        """
        if isinstance(messages, str):
            prompt = messages
        elif isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"{msg['content']}\n\n"
                elif msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"
        else:
            raise ValueError("Invalid message format for Gemini prompt.")

        return query_gemini(prompt).strip()

    def extract_config(self, user_input: str) -> dict:
        """
        Run Few-shot CoT extraction for the given user command.
        Returns a dictionary with algorithm, dataset, and parameters.
        """
        # Prepare prompt
        prompt = [dict(p) for p in self.FEW_SHOT_COT_PROMPT]
        prompt[-1]["content"] = prompt[-1]["content"].format(user_input=user_input)

        assistant_text = self.get_gemini_response(prompt)
        print("=== Gemini Response ===\n", assistant_text)

        # Extract JSON
        match = re.search(r"FINAL:\s*(\{.*\})", assistant_text, re.DOTALL | re.IGNORECASE)
        json_str = match.group(1) if match else None

        if not json_str:
            # fallback: first {...} block
            start = assistant_text.find('{')
            end = assistant_text.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = assistant_text[start:end+1]

        if not json_str:
            print("[extract_config] No JSON found in assistant response.")
            return {}

        try:
            parsed = json.loads(json_str)
        except Exception as e:
            print(f"[extract_config] JSON parse error: {e}")
            return {}

        # Clean dataset paths
        def _clean_path(v):
            if isinstance(v, str):
                return os.path.normpath(v.strip().strip('"').strip("'"))
            return v

        if parsed.get("dataset_train"):
            parsed["dataset_train"] = _clean_path(parsed["dataset_train"])
        if parsed.get("dataset_test"):
            parsed["dataset_test"] = _clean_path(parsed["dataset_test"])

        # Auto-fill algorithm if missing
        if not parsed.get("algorithm"):
            user_lower = user_input.lower()
            if ("anomaly" in user_lower or "detect" in user_lower) and parsed.get("dataset_train"):
                parsed["algorithm"] = ["all"]
                print("[INFO] No algorithm specified — auto-selecting all algorithms.")

        return parsed

    def run_chatbot(self):
        """
        Main loop: prompts user for input, extracts config, and handles missing fields.
        """
        while not all([
            self.experiment_config["dataset_train"],
            os.path.exists(self.experiment_config["dataset_train"]),
            (not self.experiment_config["dataset_test"] or os.path.exists(self.experiment_config["dataset_test"]))
        ]):
            if len(self.messages) == 1:
                print("Enter command (e.g., 'Run IForest on ./data/glass_train.mat and ./data/glass_test.mat with contamination=0.1'):")

            user_input = input("User: ").strip()
            if not user_input:
                continue

            self.messages.append({"role": "user", "content": user_input})

            extracted = self.extract_config(user_input)
            print("[DEBUG] extracted:", extracted)

            if extracted.get("algorithm"):
                self.experiment_config["algorithm"] = extracted["algorithm"]
            if extracted.get("dataset_train"):
                self.experiment_config["dataset_train"] = extracted["dataset_train"]
            if extracted.get("dataset_test"):
                self.experiment_config["dataset_test"] = extracted["dataset_test"]
            if extracted.get("parameters"):
                self.experiment_config["parameters"].update(extracted["parameters"])

            # Prompt user if dataset paths invalid
            if not os.path.exists(self.experiment_config["dataset_train"]):
                print("Chatbot: Please provide a valid training dataset location.")

        # Final output summary
        print("\nExperiment Configuration")
        print("Algorithm        :", self.experiment_config["algorithm"])
        print("Training Dataset :", self.experiment_config["dataset_train"])
        print("Testing Dataset  :", self.experiment_config["dataset_test"])
        print("Parameters       :", self.experiment_config["parameters"])


if __name__ == "__main__":
    from config.config import Config
    import google.generativeai as genai
    genai.configure(api_key=Config.GEMINI_API_KEY)

    chatbot_instance = AgentProcessor()
    chatbot_instance.run_chatbot()
