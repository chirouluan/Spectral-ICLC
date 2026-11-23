"""Minimal LUMIR classification Prompt definitions, extracted without behavioral changes."""

def format_examples(examples):
    return '\n'.join(f"{e.get('name', 'ex')} : x={e['x']}, label={e['label']}" for e in examples)


class SinglePrompt:
    SYSTEM_PROMPT = 'You are a spectral chemometrics and pattern recognition expert. Task: given few-shot examples (x and label), classify the test samples. Output MUST be a JSON array of labels with length exactly equal to the number of test samples. Do NOT include any extra text or code fences.'
    def format_eval_prompt(self, eval_examples):
        xs = [e['x'] for e in eval_examples]
        return f'Known examples are above. Now classify the following samples. Return ONLY a JSON array, e.g.: ["A","B",...]. Samples: {xs}'

class AdaptivePrompt:
    SYSTEM_PROMPT = 'You are a spectral chemometrics and pattern recognition expert. Given few-shot examples (feature vector x and label), classify the provided samples. Your OUTPUT MUST be a JSON array of labels whose length equals the number of samples. Do NOT include extra text or code fences.'
    def format_eval_prompt(self, eval_examples):
        xs = [e['x'] for e in eval_examples]
        return f'Use the known examples above to classify the following samples.\nReturn ONLY a JSON array like ["A","B",...].\nSamples: {xs}'

SOURCE_SHA256 = {'Generate_single.py': 'd1448010f1f331a268f778736fae71623daf0dce1db2e0e177b6f293bcee56f1', 'Mul_Generate.py': '8777b40688ee1016a218eb4e7cac98b5d63beaaa59c4b7fb2ecddb36b82c38f6', 'other_models.py': 'c6731fc0e8c84178bc35df28a3e3310ce01e1bda7abf66e749d5ef11c13f5a92'}
