import numpy as np


class SampleLLM:
    """This is a sample LLM which always answer a random decision for testing the env."""

    def chat(self, model, messages=None, temperature=None, top_p=None):
        """Return a random decision from LLM model."""
        available_decisions = [1, 2, 3]
        decision = np.random.choice(available_decisions)
        return f"final decision: ({decision})"
