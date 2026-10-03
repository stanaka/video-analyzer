import os
import socket

from .errors import AnalysisError


def offline_environment() -> dict[str, str]:
    return {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYANNOTE_METRICS_ENABLED": "0",
        "DO_NOT_TRACK": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


def enforce_worker_offline():
    """Set flags before model imports, and reject Python network sockets in workers.

    This is defense in depth, not an OS sandbox for native libraries. Integration
    verification should also run with the OS network disabled.
    """
    os.environ.update(offline_environment())

    def blocked(*_args, **_kwargs):
        raise AnalysisError("Network access is disabled during analysis. Run prepare-models first.")

    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_sendto = socket.socket.sendto

    def connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            return blocked()
        return original_connect(self, address)

    def connect_ex(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            return blocked()
        return original_connect_ex(self, address)

    def sendto(self, *args):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            return blocked()
        return original_sendto(self, *args)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.socket.sendto = sendto
