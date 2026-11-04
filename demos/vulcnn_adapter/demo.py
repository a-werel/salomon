from salomon.adapters.vulcnn.adapter import VulCNNAdapter

def main():
    adapter = VulCNNAdapter()
    source = "int add(int a, int b) { return a + b; }\n"
    pdgs = adapter.create_pdg(source, filename="example.c")
    report = adapter.analyze_code(source.encode("UTF-8"), filename="example.c")
    print(report.read_text(encoding="UTF-8"))

if __name__ == "__main__":
    main()