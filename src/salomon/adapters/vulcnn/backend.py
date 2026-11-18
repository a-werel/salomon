from pathlib import Path

from salomon.adapters.AbstractBackend import AbstractBackend


class VulCNNBackend(AbstractBackend):
    """AbstractBackend implementation for the VulCNN model.

    Args:
        workspace (str | Path | None): Path to workspace
        models (str | Path | None): Path to models
        image (str): Container image name, used when there is no container
        container (str | None): Name or ID of existing container
        embedding_model (str): Relative path to the embedding model in models/ project directory
        nn_model (str): Relative path to the neural network in the models/ project directory
        max_length (int): Max length of each function's feature representation
        timeout (int): Maximum number of seconds that the container can run.
    """

    def __init__(
        self,
        workspace: str | Path | None = None,
        models: str | Path | None = None,
        image: str = "salomon-vulcnn:local",
        container: str | None = None,
        embedding_model: str = "vulcnn/sent2vec.bin",
        nn_model: str = "vulcnn/cnn.pt",
        max_length: int = 100,
        timeout: int = 600,
    ):
        super().__init__(
            workspace=workspace,
            models=models,
            image=image,
            container=container,
            embedding_model=embedding_model,
            nn_model=nn_model,
            max_length=max_length,
            timeout=timeout,
            workspace_environment_variable="VULCNN_WORKSPACE",
            models_environment_variable="VULCNN_MODELS",
            container_environment_variable="VULCNN_CONTAINER",
        )

        # The wheel includes this same script; source checkouts use docker/vulcnn/cli.py.
        self.cli_absolute_path = Path(__file__).resolve().parents[2] / "vulcnn_cli.py"
        if not self.cli_absolute_path.is_file():
            self.cli_absolute_path = (
                Path(__file__).resolve().parents[4] / "docker/vulcnn/cli.py"
            )

    def check_if_models_are_present_in_project_dir(self) -> None:
        for model in (
            self.embedding_model_relative_path,
            self.nn_model_relative_path,
        ):
            if not (self.models_absolute_path / model).is_file():
                raise FileNotFoundError(f"Required trained model is missing: {self.models_absolute_path / model}")

    def create_model_representation(self, run: Path) -> dict:
        for stage in ("inspect", "normalize", "parse", "export"):
            result = self.run_model_command(
                stage,
                run=self.host_to_container_workspace_path(run),
            )
        return result
