import sys
import os
import re
import time
import copy
import types
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, root_mean_squared_error, accuracy_score

# Import jolt-main modules
_jolt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'jolt-main')
if _jolt_dir not in sys.path:
    sys.path.insert(0, _jolt_dir)

from helpers import (
    process_generated_results,
    get_dimension,
)


def convert_to_jolt_format(train_data, test_data, task_type,
                           feature_names=None, num_decimal_x=4):
    """
    Convert main pipeline's list[dict] to JoLT's data format.

    Args:
        train_data: list[dict], each with "x" (list[float]) and "label"/"y"
        test_data: list[dict], same format
        task_type: "classification" or "regression"
        feature_names: optional list of feature column names
        num_decimal_x: decimal places for x values in string format

    Returns:
        data: dict with x_train, y_train, x_test, y_test as numpy arrays
        y_column_types: list[str], e.g. ["categorical"] or ["numerical"]
        y_column_names: list[str], e.g. ["class"] or ["value"]
    """
    n_features = len(train_data[0]["x"])
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    sep = ";"
    name_val_sep = ":"

    def format_x(x_vec):
        parts = []
        for name, val in zip(feature_names, x_vec):
            parts.append(f"{name}{name_val_sep}{val:.{num_decimal_x}f}")
        return sep.join(parts) + sep

    x_train = np.array([format_x(d["x"]) for d in train_data])
    x_test = np.array([format_x(d["x"]) for d in test_data])

    if task_type == "classification":
        y_train = np.array([[d["label"]] for d in train_data], dtype=object)
        y_test = np.array([[d["label"]] for d in test_data], dtype=object)
        y_column_types = ["categorical"]
        y_column_names = ["class"]
    elif task_type == "regression":
        y_train = np.array([[d["y"]] for d in train_data], dtype=object)
        y_test = np.array([[d["y"]] for d in test_data], dtype=object)
        y_column_types = ["numerical"]
        y_column_names = ["value"]
    else:
        raise ValueError(f"JoLT does not support task type: {task_type}")

    data = {
        "x_train": x_train, "y_train": y_train,
        "x_test": x_test, "y_test": y_test,
        "x_true": np.concatenate([x_train, x_test]),
        "y_true": np.concatenate([y_train, y_test]),
    }
    return data, y_column_types, y_column_names


def build_jolt_args(task_type, y_column_types, y_column_names,
                    num_samples=10, temperature=1.0, batch_size=5,
                    max_generated_length=20, num_decimal_places_x=4,
                    num_decimal_places_y=2, output_dir="./output",
                    experiment_name="jolt_experiment", llm_type="api",
                    mode="sample_only"):
    """
    Build a minimal args Namespace for JoLT functions.
    Avoids parse_args.py and its jsonargparse dependency.
    """
    return types.SimpleNamespace(
        # Data format
        prefix="",
        break_str="\n",
        column_separator=";",
        name_value_separator=":",
        header_option="headers_as_item_prefix",
        y_column_names=y_column_names,
        y_column_types=y_column_types,
        num_decimal_places_x=num_decimal_places_x,
        num_decimal_places_y=num_decimal_places_y,
        # Sampling params
        num_samples=num_samples,
        temperature=temperature,
        top_p=0.9,
        top_k=None,
        batch_size=batch_size,
        max_generated_length=max_generated_length,
        # Output
        output_dir=output_dir,
        experiment_name=experiment_name,
        # Debug
        print_prompts=False,
        print_sampling_rejections=True,
        # Model identifier
        llm_type=llm_type,
        llm_path=None,
        # Mode
        mode=mode,
        # CSV parser fields (unused but present for compatibility)
        data_path=None,
        csv_split_option="test_fraction",
        test_fraction=0.2,
        shots=None,
        train_size_limit=None,
        test_size_limit=None,
        train_start_index=None,
        train_end_index=None,
        test_start_index=None,
        test_end_index=None,
        missing_fraction=0.0,
        impute_features=False,
        shuffle=True,
        seed=1,
        columns_to_ignore=[],
        # Quantization
        quantize_4bit=False,
    )


def _extract_categories(data, y_column_types):
    """
    Extract category names from y data for JoLT's categorical handling.
    Returns list[list[str]], one list per y column.
    """
    categories = []
    for col_idx, col_type in enumerate(y_column_types):
        if col_type == "categorical":
            y_all = np.concatenate([data["y_train"][:, col_idx],
                                    data["y_test"][:, col_idx]])
            cats = sorted(set(str(v) for v in y_all))
            categories.append(cats)
        else:
            categories.append([])
    return categories


def _parse_llm_output(text, task_type, categories):
    """
    Parse LLM output in main pipeline's Python list format.
    Expected: [y1, y2, y3, ...] or single value.
    Returns list of parsed values, or empty list if extraction fails.
    """
    text = text.strip()
    if not text:
        return []

    # Remove ```python``` markers
    clean = text.replace("```python\n", "").replace("\n```", "").strip()

    if task_type == "regression":
        # Try Python list format: [y1, y2, y3, ...]
        if clean.startswith("[") and clean.endswith("]"):
            inner = clean.strip("[]")
            str_tokens = [tok.strip().strip("'\"") for tok in inner.split(",")]
            float_preds = [
                float(tok) for tok in str_tokens
                if re.match(r"^-?\d+(\.\d+)?(e[+-]?\d+)?$", tok, re.IGNORECASE)
            ]
            if float_preds:
                return float_preds

        # Fallback: try single number
        try:
            return [float(clean)]
        except ValueError:
            pass

        # Fallback: extract numbers from text
        numbers = re.findall(r'-?\d+\.?\d*(?:e[+-]?\d+)?', clean)
        if numbers:
            return [float(n) for n in numbers]

        return []
    else:  # classification
        results = []
        # Try Python list format
        if clean.startswith("[") and clean.endswith("]"):
            inner = clean.strip("[]")
            tokens = [tok.strip().strip("'\"") for tok in inner.split(",")]
            for tok in tokens:
                if tok:
                    results.append(tok)
            return results

        # Fallback: single value
        return [clean] if clean else []


def _build_prompt_main_style(train_data_raw, test_x_list, task_type):
    """
    Build prompt using the same format as SpectrumReg (main pipeline).

    Args:
        train_data_raw: list[dict] with 'x', 'label'/'y' keys
        test_x_list: list of x vectors (list[list[float]])
        task_type: "regression" or "classification"

    Returns:
        messages: list of {"role": ..., "content": ...} for chat API
    """
    # System prompt — same as SpectrumReg
    system_msg = (
        "You are a spectral chemometrics and regression expert. "
        "Given examples of (x, y), predict y for new x values. "
        "Output ONLY a Python list. No explanation, no code."
    )

    # Training data — same format as SpectrumReg.format_read()
    train_msg = "Training data (name, x, y):\n" + repr(train_data_raw)

    # Test data — same format as SpectrumReg.format_predict()
    n = len(test_x_list)
    if task_type == "regression":
        test_msg = (
            f"Predict y for {n} test samples. Output ONLY a Python list of {n} floats.\n"
            f"Test x values:\n{test_x_list}\n"
            f"Output format: [y1, y2, y3, ... y{n}]"
        )
    else:
        test_msg = (
            f"Predict class for {n} test samples. Output ONLY a Python list of {n} class names.\n"
            f"Test x values:\n{test_x_list}\n"
            f"Output format: [class1, class2, ... class{n}]"
        )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": train_msg},
        {"role": "user", "content": test_msg},
    ]


def _api_generate(client, model_name, messages, batch_size,
                  temperature, top_p, max_new_tokens,
                  task_type="regression", categories=None,
                  max_retries=5, base_delay=2.0, call_interval=0.5,
                  debug=False):
    """
    Generate multiple samples using OpenAI-compatible chat API.
    Uses main pipeline's prompt format (system + training + test messages).
    """
    results = []

    for i in range(batch_size):
        if i > 0:
            time.sleep(call_interval)
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                )
                text = resp.choices[0].message.content
                if debug and i == 0:
                    print(f"[JoLT DEBUG] API raw response: {repr(text[:300])}")
                if text is None:
                    print("[JoLT] API returned None content")
                    continue
                text = text.strip()
                results.append(text)
                break
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate" in err_str.lower()
                if is_rate_limit and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"[JoLT] Rate limited, retrying in {delay:.1f}s "
                          f"(attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    print(f"[JoLT] API call failed: {e}")
                    break
    return results


def _parse_batch_output(text, n_expected, task_type, categories):
    """Parse Python list output [y1, y2, ...] into individual values."""
    return _parse_llm_output(text, task_type, categories)


def _run_api(data, args, client, model_name, call_interval=0.5, test_batch_size=1,
             train_data_raw=None, test_data_raw=None):
    """
    Run JoLT sampling using OpenAI-compatible chat API.
    Uses main pipeline's prompt format (SpectrumReg style).
    """
    dim_y = len(args.y_column_types)
    categories = _extract_categories(data, args.y_column_types)
    task_type = "regression" if args.y_column_types[0] == "numerical" else "classification"

    n_test = len(data["x_test"])

    # Build prompts using main pipeline format (use raw data, not JoLT-converted)
    if test_batch_size <= 1:
        # Single-sample mode: one test sample per prompt
        prompts_messages = []
        for i in range(n_test):
            test_x = [test_data_raw[i]["x"]]
            messages = _build_prompt_main_style(train_data_raw, test_x, task_type)
            prompts_messages.append(messages)

        if prompts_messages:
            print(f"[JoLT DEBUG] Prompt preview (system + user messages):")
            for msg in prompts_messages[0]:
                print(f"  [{msg['role']}] {msg['content'][:150]}...")

        gen = [[] for _ in range(n_test)]
        for idx, messages in enumerate(tqdm(prompts_messages, desc="JoLT API Sampling")):
            if idx > 0:
                time.sleep(call_interval)
            samples = []
            remaining = args.num_samples
            attempts = 0
            consecutive_fails = 0
            max_total_retries = args.num_samples * 3

            while remaining > 0 and attempts < max_total_retries:
                attempts += 1
                bs = min(args.batch_size, remaining)
                gen_texts = _api_generate(
                    client=client, model_name=model_name, messages=messages,
                    batch_size=bs, temperature=args.temperature, top_p=args.top_p,
                    max_new_tokens=args.max_generated_length,
                    task_type=task_type, categories=categories,
                    debug=(idx == 0),
                )
                batch_had_success = False
                for text in gen_texts:
                    parsed_list = _parse_llm_output(text, task_type, categories)
                    if idx == 0 and len(samples) == 0:
                        print(f"[JoLT DEBUG] Raw: {repr(text[:200])} -> {parsed_list}")
                    for val in parsed_list:
                        if remaining > 0:
                            samples.append(np.array([val], dtype=object))
                            remaining -= 1
                            batch_had_success = True
                if batch_had_success:
                    consecutive_fails = 0
                else:
                    consecutive_fails += 1
                    if consecutive_fails >= 3:
                        break
            gen[idx] = samples
    else:
        # Batched mode: multiple test samples per prompt
        n_batches = (n_test + test_batch_size - 1) // test_batch_size
        print(f"[JoLT] Batched mode: {n_test} test samples, batch_size={test_batch_size}, "
              f"{n_batches} prompts × {args.num_samples} samples = {n_batches * args.num_samples} calls")

        gen = [[] for _ in range(n_test)]
        for batch_idx in tqdm(range(n_batches), desc="JoLT Batched Sampling"):
            start = batch_idx * test_batch_size
            end = min(start + test_batch_size, n_test)
            batch_indices = list(range(start, end))
            batch_x = [test_data_raw[i]["x"] for i in batch_indices]

            messages = _build_prompt_main_style(train_data_raw, batch_x, task_type)
            if batch_idx == 0:
                print(f"[JoLT DEBUG] Batch prompt preview:")
                for msg in messages:
                    print(f"  [{msg['role']}] {msg['content'][:200]}...")

            all_batch_samples = {i: [] for i in batch_indices}
            remaining = {i: args.num_samples for i in batch_indices}
            attempts = 0
            consecutive_fails = 0
            max_total_retries = args.num_samples * 3

            while any(r > 0 for r in remaining.values()) and attempts < max_total_retries:
                attempts += 1
                if batch_idx > 0 or attempts > 1:
                    time.sleep(call_interval)

                gen_texts = _api_generate(
                    client=client, model_name=model_name, messages=messages,
                    batch_size=1, temperature=args.temperature, top_p=args.top_p,
                    max_new_tokens=max_generated_length_for_batch(len(batch_indices), task_type),
                    task_type=task_type, categories=categories,
                    debug=(batch_idx == 0 and attempts == 1),
                )

                batch_had_success = False
                for text in gen_texts:
                    parsed_list = _parse_batch_output(text, len(batch_indices), task_type, categories)
                    if batch_idx == 0 and attempts == 1:
                        print(f"[JoLT DEBUG] Batch output: {repr(text[:300])} -> {len(parsed_list)} values")
                    for j, val in enumerate(parsed_list):
                        if j < len(batch_indices) and remaining[batch_indices[j]] > 0:
                            all_batch_samples[batch_indices[j]].append(np.array([val], dtype=object))
                            remaining[batch_indices[j]] -= 1
                            batch_had_success = True

                if batch_had_success:
                    consecutive_fails = 0
                else:
                    consecutive_fails += 1
                    if consecutive_fails >= 3:
                        break

            for i in batch_indices:
                gen[i] = all_batch_samples[i]

    empty_count = sum(1 for s in gen if len(s) == 0)
    if empty_count > 0:
        print(f"[JoLT] WARNING: {empty_count}/{n_test} test points have 0 valid samples")

    results = {
        "data": data,
        "args": args,
        "dim_x": get_dimension(data["x_train"]),
        "dim_y": dim_y,
        "categories": categories,
        "gen": gen,
    }
    results = process_generated_results(gen_results=results, args=args)
    return results


def max_generated_length_for_batch(n_samples, task_type):
    """Estimate max tokens needed for batched output."""
    if task_type == "regression":
        return n_samples * 15  # ~15 tokens per number
    else:
        return n_samples * 10  # ~10 tokens per class name


def _run_local(data, args, llm_type, llm_path=None):
    """
    Run JoLT using local HuggingFace model via hf_api.py.
    """
    from hf_api import get_model_and_tokenizer
    from sample import sample as jolt_sample

    args.llm_type = llm_type
    if llm_path:
        args.llm_path = llm_path

    model, tokenizer = get_model_and_tokenizer(args)

    dim_y = len(args.y_column_types)
    categories = _extract_categories(data, args.y_column_types)

    results = {
        "data": data,
        "args": args,
        "dim_x": get_dimension(data["x_train"]),
        "dim_y": dim_y,
        "categories": categories,
    }
    results = jolt_sample(args, tokenizer, model, results)
    return results


def compute_jolt_metrics(results, task_type):
    """
    Extract metrics from JoLT results aligned with main pipeline format.

    Returns:
        dict with metric names and values
    """
    metrics = {}

    if task_type == "regression":
        y_true = np.array(results["data"]["y_test"], dtype=float).flatten()
        y_pred = np.array(results["metrics"][0]["y_test_median"], dtype=float)
        # Filter out NaN predictions (from test points with 0 valid samples)
        valid = ~np.isnan(y_pred)
        if valid.sum() < 2:
            print("[JoLT] Not enough valid predictions to compute metrics")
            return metrics
        y_true_v, y_pred_v = y_true[valid], y_pred[valid]
        metrics["r2"] = round(r2_score(y_true_v, y_pred_v), 4)
        metrics["rmse"] = round(root_mean_squared_error(y_true_v, y_pred_v), 2)
        metrics["mae"] = round(results["metrics"][0]["mae"], 4)
        if valid.sum() < len(y_pred):
            print(f"[JoLT] Metrics computed on {valid.sum()}/{len(y_pred)} valid predictions")

    elif task_type == "classification":
        y_true = np.array(results["data"]["y_test"]).flatten()
        y_pred = results["metrics"][0]["y_test_mode"]
        # Filter out NaN predictions
        valid = [i for i, v in enumerate(y_pred) if not (isinstance(v, float) and np.isnan(v))]
        if len(valid) < 2:
            print("[JoLT] Not enough valid predictions to compute metrics")
            return metrics
        y_true_v = y_true[valid]
        y_pred_v = [y_pred[i] for i in valid]
        metrics["accuracy"] = round(accuracy_score(y_true_v, y_pred_v), 4)
        if "auc_sampling" in results["metrics"][0]:
            metrics["auc"] = round(results["metrics"][0]["auc_sampling"], 4)
        if len(valid) < len(y_pred):
            print(f"[JoLT] Metrics computed on {len(valid)}/{len(y_pred)} valid predictions")

    return metrics


class JoltRunner:
    """
    JoLT integration runner for the main spectral analysis pipeline.

    Adapts main pipeline's train_data/test_data (list[dict]) to JoLT format,
    runs JoLT sampling via API or local HuggingFace model, and returns
    metrics aligned with the main pipeline's output format.

    Usage:
        runner = JoltRunner(mode="api")
        result = runner.run(
            train_data=train_data, test_data=test_data,
            task_type="regression",
            client=OpenAI(api_key="...", base_url="..."),
            model_name="mimo-v2.5-pro",
        )
        print(result["metrics"])  # {"r2": 0.75, "rmse": 0.31, "mae": 0.22}
    """

    def __init__(self, mode="api"):
        """
        Args:
            mode: "api" for OpenAI-compatible API, "local" for HuggingFace model
        """
        if mode not in ("api", "local"):
            raise ValueError(f"mode must be 'api' or 'local', got '{mode}'")
        self.mode = mode

    def run(self, train_data, test_data, task_type,
            client=None, model_name=None,
            llm_type=None, llm_path=None,
            num_samples=10, temperature=1.0, batch_size=5,
            max_generated_length=20, num_decimal_places_y=2,
            call_interval=0.5, test_batch_size=1):
        """
        Run JoLT inference on the given data.

        Args:
            train_data: list[dict] from main pipeline's dataset stage
            test_data: list[dict] from main pipeline's dataset stage
            task_type: "classification" or "regression"
            client: OpenAI client instance (required for api mode)
            model_name: API model name (required for api mode)
            llm_type: HuggingFace model type (required for local mode)
            llm_path: HuggingFace model path (optional for local mode)
            num_samples: number of samples per test point
            temperature: sampling temperature
            batch_size: samples per API call / generation batch
            max_generated_length: max tokens to generate
            num_decimal_places_y: decimal places for y values
            call_interval: seconds to wait between API calls (rate limit)
            test_batch_size: number of test samples per prompt (1=original, >1=batched)

        Returns:
            dict with keys:
                "results": raw JoLT results dict
                "metrics": aligned metrics dict (r2/rmse/mae or accuracy/auc)
                "prompts": list of prompt strings used
        """
        if task_type == "anomaly detection":
            print("[JoLT] Anomaly detection is not supported. Skipping.")
            return {"results": None, "metrics": {}, "prompts": []}

        # Save copies of original numeric data before convert_to_jolt_format modifies them
        train_data_raw = copy.deepcopy(train_data)
        test_data_raw = copy.deepcopy(test_data)

        # Convert data format
        data, y_column_types, y_column_names = convert_to_jolt_format(
            train_data, test_data, task_type
        )

        # Build args
        args = build_jolt_args(
            task_type=task_type,
            y_column_types=y_column_types,
            y_column_names=y_column_names,
            num_samples=num_samples,
            temperature=temperature,
            batch_size=batch_size,
            max_generated_length=max_generated_length,
            num_decimal_places_y=num_decimal_places_y,
        )

        # Run inference
        if self.mode == "api":
            if client is None or model_name is None:
                raise ValueError("API mode requires 'client' and 'model_name'")
            print(f"[JoLT] Running API mode with model={model_name}, "
                  f"num_samples={num_samples}, temperature={temperature}, "
                  f"test_batch_size={test_batch_size}, call_interval={call_interval}s")
            results = _run_api(data, args, client, model_name, call_interval, test_batch_size,
                               train_data_raw=train_data_raw, test_data_raw=test_data_raw)
        else:
            if llm_type is None:
                raise ValueError("Local mode requires 'llm_type'")
            print(f"[JoLT] Running local mode with llm_type={llm_type}, "
                  f"num_samples={num_samples}, temperature={temperature}")
            results = _run_local(data, args, llm_type, llm_path)

        # Compute aligned metrics
        metrics = compute_jolt_metrics(results, task_type)
        print(f"[JoLT] Metrics: {metrics}")

        return {
            "results": results,
            "metrics": metrics,
            "prompts": results.get("prompts", []),
        }
