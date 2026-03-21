import torch
from torchinfo import summary


def get_model_complexity(model, input_size, device='cpu'):
    """
    Calculate GFlops and Params using torchinfo.

    Args:
        model: PyTorch model
        input_size: Tuple including batch dim, e.g. (1, 3, 1024, 1024)
        device: Device string
    """
    try:
        stats = summary(model, input_size=input_size, verbose=0, device=device)

        params = stats.total_params
        macs = stats.total_mult_adds
        gflops = 2 * macs / 1e9

        return {
            'Params (M)': params / 1e6,
            'GFlops': gflops,
            'MACs (G)': macs / 1e9,
        }

    except Exception as e:
        print(f"Complexity calculation failed: {e}")
        return {
            'Params (M)': 0,
            'GFlops': 0,
            'MACs (G)': 0,
        }
