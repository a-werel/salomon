from abc import ABC, abstractmethod
from pathlib import Path


class AbstractAdapter(ABC):

    @abstractmethod
    def create_representation(self, source_code: str | bytes, filename: str) -> list[Path]:
        """ 
        Creates model specific representation of source code, i.e. PDG, CPG, AST, etc.
        Returns a list of paths to created representations.
        """

    @abstractmethod
    def analyze_code(self, source_code: str | bytes, filename: str) -> Path:
        """ 
        Creates a classification report .txt file.
        Returns the path to the classification report.
        """