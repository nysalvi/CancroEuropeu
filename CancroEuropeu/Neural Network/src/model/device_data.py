from dataclasses import dataclass

@dataclass
class DeviceData:
    isCuda: bool
    device: str

    def __init__(self, isCuda: bool, device: str) -> None:
        self.isCuda = isCuda
        self.device = device
        pass