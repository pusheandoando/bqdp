# bqdp/nn/device.py
import torch





def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    metal = getattr(torch.backends, "mps", None)
    if metal is not None and metal.is_available():
        return torch.device("mps")
    
    return torch.device("cpu")