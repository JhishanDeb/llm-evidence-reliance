"""Stand-in for torch in the CPU test of the engineering check and the smoke test (only torch.cuda.device_count
is used there). Offline tests only."""
__version__ = "0.0-fake"


class cuda:
    @staticmethod
    def device_count():
        return 0


class Tensor:          # only so that libraries which check isinstance(x, torch.Tensor) keep working
    pass
