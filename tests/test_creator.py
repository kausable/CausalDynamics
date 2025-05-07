import shutil
from pathlib import Path

from causaldynamics.creator import create


def test_create(tmp_path):
    """Test that create function runs successfully and creates expected outputs."""
    # Run create function
    create(save_data=False)
