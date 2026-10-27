class VulCNNAdapter:
    def __init__(self):
        pass

    def generate_pdg(self, file_path: str) -> None:
        pass

    def predict(self, pdgs: list[PDG]) -> bool:
        pass

    def analyze_code(self, file_path: str) -> None:
        pdg = self.generate_pdg(file_path)
        self.predict(pdg)