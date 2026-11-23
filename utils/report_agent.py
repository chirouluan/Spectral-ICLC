"""
Spectral Report Agent.
LLM-generated structured analysis report (JSON + Markdown).

Ported from TimeSeriesScientist spectral_report_agent.py,
adapted to use OpenAI SDK instead of LangChain.
"""

import json
import re
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from openai import OpenAI

logger = logging.getLogger(__name__)

REPORT_SYSTEM_PROMPT = """You are the Report Generation Agent for an advanced spectral analysis system.
Your mission is to generate comprehensive, well-structured reports that summarize the entire analysis workflow.

You have expertise in scientific reporting and data visualization interpretation.
You can synthesize complex technical information into clear, actionable insights.

Report structure:
1. Executive Summary
2. Data Overview
3. Preprocessing Results
4. Model Selection
5. Performance Metrics
6. Key Findings
7. Recommendations
8. Limitations"""


class SpectralReportAgent:
    """
    LLM-based spectral analysis report generator.
    Collects results from all pipeline stages and generates a structured report.
    """

    def __init__(self, api_key: str, base_url: str, model_name: str, client=None):
        self.client = client if client is not None else OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def run(
        self,
        analysis_result: Dict[str, Any],
        selected_methods: Dict[str, Any],
        model_results: Dict[str, Any],
        task_type: str = "quantitative",
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive spectral analysis report.

        Args:
            analysis_result: output from SpectralAnalysisAgent.run()
            selected_methods: preprocessing + feature extraction methods used
            model_results: model training/evaluation results
            task_type: "quantitative", "classification", or "anomaly detection"
            output_dir: directory to save report files (optional)

        Returns:
            Report dict with keys: executive_summary, data_overview,
            preprocessing_results, model_selection, performance_metrics,
            key_findings, recommendations, limitations
        """
        logger.info("Generating comprehensive report...")

        report_data = {
            "task_type": task_type,
            "analysis": analysis_result,
            "methods": selected_methods,
            "model_performance": model_results,
        }

        prompt = f"""Generate a comprehensive spectral analysis report based on the following data.

## Task Type: {task_type}

## Methods Used
{json.dumps(selected_methods, indent=2, default=str)}

## Data Analysis
{json.dumps({k: v for k, v in analysis_result.items() if k != 'raw_response'}, indent=2, default=str)}

## Model Performance
{json.dumps(model_results, indent=2, default=str)}

Return a structured JSON report:
{{
    "executive_summary": "Brief overview (2-3 sentences)",
    "data_overview": "Summary of spectral data characteristics",
    "preprocessing_results": "What preprocessing was applied and why",
    "model_selection": "Which models were selected and why",
    "performance_metrics": "Detailed performance analysis",
    "key_findings": ["finding 1", "finding 2"],
    "recommendations": ["recommendation 1", "recommendation 2"],
    "limitations": ["limitation 1", "limitation 2"]
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            msg = response.choices[0].message
            text = (msg.content or "").strip()
            if not text and hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                text = msg.reasoning_content.strip()
            import re
            # Extract answer after <think>...</think> if present
            think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
            after_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            if after_think:
                text = after_think
            elif think_match:
                think_content = think_match.group(1).strip()
                json_match = re.search(r"\{.*?\}", think_content, flags=re.DOTALL)
                if json_match:
                    text = json_match.group(0)
                else:
                    text = think_content
            report = self._parse_report(text)
            logger.info("Report generated via LLM")
        except Exception as e:
            logger.warning(f"LLM report generation failed: {e}, using fallback")
            report = self._fallback_report(report_data, task_type)

        # Save if output_dir specified
        if output_dir:
            self._save_report(report, output_dir, task_type)

        return report

    def _parse_report(self, text: str) -> Dict[str, Any]:
        """Parse LLM report response."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        m = re.search(r"```(?:json|python)?\s*(.*?)```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return {
            "executive_summary": text,
            "data_overview": "",
            "preprocessing_results": "",
            "model_selection": "",
            "performance_metrics": "",
            "key_findings": [],
            "recommendations": [],
            "limitations": [],
        }

    def _fallback_report(self, report_data: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Deterministic fallback report."""
        perf = report_data.get("model_performance", {})
        best = perf.get("best_model", "N/A")
        return {
            "executive_summary": f"Spectral {task_type} analysis completed. Best model: {best}.",
            "data_overview": "See analysis_result for details.",
            "preprocessing_results": f"Methods: {report_data.get('methods', {})}",
            "model_selection": f"Best model: {best}",
            "performance_metrics": json.dumps(perf, indent=2, default=str),
            "key_findings": [f"Best performing model: {best}"],
            "recommendations": ["Consider fine-tuning hyperparameters"],
            "limitations": ["Analysis based on available data"],
        }

    def _save_report(self, report: Dict[str, Any], output_dir: str, task_type: str):
        """Save report as JSON and Markdown."""
        out_path = Path(output_dir) / "reports"
        out_path.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = out_path / f"spectral_{task_type}_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        # Markdown
        md_path = out_path / f"spectral_{task_type}_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Spectral {task_type.title()} Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            sections = [
                ("Executive Summary", "executive_summary"),
                ("Data Overview", "data_overview"),
                ("Preprocessing Results", "preprocessing_results"),
                ("Model Selection", "model_selection"),
                ("Performance Metrics", "performance_metrics"),
            ]
            for title, key in sections:
                if report.get(key):
                    f.write(f"## {title}\n\n{report[key]}\n\n")

            for key, title in [("key_findings", "Key Findings"), ("recommendations", "Recommendations"), ("limitations", "Limitations")]:
                if report.get(key):
                    f.write(f"## {title}\n\n")
                    for item in report[key]:
                        f.write(f"- {item}\n")
                    f.write("\n")

        logger.info(f"Report saved to {out_path}")
