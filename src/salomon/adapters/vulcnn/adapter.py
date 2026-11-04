from pathlib import Path
from salomon.adapters.AbstractAdapter import AbstractAdapter
from .backend import VulCNNBackend


class VulCNNAdapter(AbstractAdapter):
    def __init__(self, **settings):
        self.backend = VulCNNBackend(**settings)

    def create_representation(self, source_code: str | bytes, filename: str) -> list[Path]:
        run = self.backend.prepare_source(source_code, filename)
        manifest = self.backend.create_pdg(run)
        return [self.backend.host_path(function["pdg"]) for function in manifest["functions"]]

    def create_pdg(self, source_code: str | bytes, filename: str = "source.c") -> list[Path]:
        return self.create_representation(source_code, filename)

    def analyze_code(self, source_code: str | bytes, filename: str = "source.c") -> Path:
        self.backend.check_models()
        run = self.backend.prepare_source(source_code, filename)
        self.backend.create_pdg(run)
        self.backend.run_vulcnn_command(
            "embed", run=self.backend.container_path(run), model=f"/models/{self.backend.embedding_model}"
        )
        result = self.backend.run_vulcnn_command(
            "classify",
            run=self.backend.container_path(run),
            nn_model=f"/models/{self.backend.nn_model}",
            max_length=self.backend.max_length,
        )
        return self.backend.host_path(result["report"])
