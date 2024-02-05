# Import version and other metadata, you might not have these exactly but can add what's relevant
from LymphoMNIST.info import __version__, HOMEPAGE

# Attempt to import your main class(es) and any utility functions or classes
try:
    from LymphoMNIST.LymphoMNIST import LymphoMNIST
    # Import any other classes or functions you have defined and want to expose directly
    from .utils import plot_dl
except ImportError as e:
    print(f"An error occurred: {e}.")
    print("Please install the required packages first. " +
          "Use `pip install -r requirements.txt`.")

