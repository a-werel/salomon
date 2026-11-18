""" 
Define the stages of the VulCNN pipeline:
1. Inspect: Inspect the source code and generate a CPG.
2. Normalize: Normalize the source code and generate a normalized CPG. (remove commands, normalize identifiers)
3. Parse: Parse the normalized CPG and generate a PDG.
4. Export: Export the PDG as JSON files.
5. Embed: Embed the PDG into a feature vector.
6. Classify: Classify the feature vector into either vulnerable or non-vulnerable.
"""

STAGES = ("inspect", "normalize", "parse", "export", "embed", "classify")

"""
Define the upstream scripts that are used in the VulCNN pipeline:
- normalization.py: Normalize the source code and generate a normalized CPG. (remove commands, normalize identifiers)
- ImageGeneration.py: Embed the PDG into a feature vector.
- generate_train_test_data.py: Generate the training and test data for the VulCNN model.
- VulCNN.py: Classify the feature vector into either vulnerable or non-vulnerable.
"""
UPSTREAM_SCRIPTS = ("normalization.py", "ImageGeneration.py", "generate_train_test_data.py", "VulCNN.py")