"""Module entry point so `python -m neurocomplexity ...` works."""
import sys
from neurocomplexity.cli import main


if __name__ == "__main__":
    sys.exit(main())
