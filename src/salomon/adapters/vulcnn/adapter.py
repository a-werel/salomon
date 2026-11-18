from pathlib import Path
from salomon.adapters.AbstractAdapter import AbstractAdapter
from .backend import VulCNNBackend


class VulCNNAdapter(AbstractAdapter):
    def __init__(self, **settings):
        self.backend = VulCNNBackend(**settings)

    def create_representation(self, source_code: str | bytes, filename: str) -> list[Path]:
        run = self.backend.prepare_original_source_code(source_code, filename)
        manifest = self.backend.create_model_representation(run)
        return [
            self.backend.container_to_host_workspace_path(function["pdg"])
            for function in manifest["functions"]
        ]

    def create_pdg(self, source_code: str | bytes, filename: str = "source.c") -> list[Path]:
        return self.create_representation(source_code, filename)

    def analyze_code(self, source_code: str | bytes, filename: str = "source.c") -> Path:
        self.backend.check_if_models_are_present_in_project_dir()
        run = self.backend.prepare_original_source_code(source_code, filename)
        self.backend.create_model_representation(run)
        self.backend.run_model_command(
            "embed",
            run=self.backend.host_to_container_workspace_path(run),
            model=f"/models/{self.backend.embedding_model_relative_path}",
        )
        result = self.backend.run_model_command(
            "classify",
            run=self.backend.host_to_container_workspace_path(run),
            nn_model=f"/models/{self.backend.nn_model_relative_path}",
            max_length=self.backend.max_length,
        )
        return self.backend.container_to_host_workspace_path(result["report"])
