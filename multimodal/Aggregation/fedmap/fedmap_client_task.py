# fedmap/fedmap_client_task.py
from fedops.client.app import FLClientTask

class FedMAPClientTask(FLClientTask):
    """
    Thin wrapper over FedOps FLClientTask:
    - Accepts and forwards a custom metadata_fn into registration
    - Otherwise uses FedOps lifecycle as-is
    """
    def __init__(self, cfg, registration, metadata_fn=None):
        super().__init__(cfg, registration)
        self._fedmap_metadata_fn = metadata_fn

    def _build_registration(self):
        reg = super()._build_registration()
        if self._fedmap_metadata_fn is not None:
            reg["metadata_fn"] = self._fedmap_metadata_fn
        return reg
