import scipy.io
from pathlib import Path

# Paths
project_root = Path(__file__).resolve().parents[1]
file_path = project_root / "data" / "raw" / "B0005.mat"

print("Loading:", file_path)

# Load MATLAB .mat file using SciPy
data = scipy.io.loadmat(
    file_path,
    squeeze_me=True,
    struct_as_record=False
)

print("MAT file loaded successfully.")
