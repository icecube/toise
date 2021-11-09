import pytest
import numpy as np

# hacky way to inject a helper function into tests without
# creating a separate package
@pytest.fixture(scope="session")
def round_to_scale():
    def round_to_scale(target, decimals: int):
        """
        syrupy's snapshot compares string representations, and so can't be used with pytest.approx
        Emulate by rounding target to the given number of digits in exponential notation
        """
        if isinstance(target, np.ndarray):
            scale = 10 ** np.round(np.log10(target) - 0.5)
            return np.round(target / scale, decimals) * scale
        elif isinstance(target, dict):
            return {k: round_to_scale(v, decimals) for k,v in target.items()}
        else:
            return round_to_scale(np.asarray(target), decimals)

    return round_to_scale