"""
Test cases for the banner version lookup in dna_factory/training_runner.py

The banner version is read from pyproject.toml (single source of truth).
"""

import logging
import sys
import tomllib
from pathlib import Path

# Add parent directory to path to import dna_factory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dna_factory.training_runner import (
    get_dna_factory_version,
    print_dna_factory_banner,
)


class TestGetDnaFactoryVersion:
    """Test cases for get_dna_factory_version function"""

    def test_reads_version_from_pyproject(self, tmp_path):
        """Version comes from the [project] version in pyproject.toml"""
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "dummy"\nversion = "9.9.9"\n'
        )
        script = tmp_path / "train.py"
        script.touch()
        assert get_dna_factory_version(str(script)) == "9.9.9"

    def test_walks_up_to_find_pyproject(self, tmp_path):
        """Lookup walks up from scripts in subdirectories"""
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "dummy"\nversion = "9.9.9"\n'
        )
        script = tmp_path / "sub" / "dir" / "train.py"
        script.parent.mkdir(parents=True)
        script.touch()
        assert get_dna_factory_version(str(script)) == "9.9.9"

    def test_matches_repo_pyproject(self):
        """Repo entry points resolve to the version in the repo pyproject.toml"""
        repo_root = Path(__file__).parent.parent
        with open(repo_root / "pyproject.toml", "rb") as f:
            expected = tomllib.load(f)["project"]["version"]
        assert get_dna_factory_version(str(repo_root / "sft.py")) == expected

    def test_returns_none_without_pyproject(self, tmp_path):
        """No pyproject.toml anywhere up the tree yields None"""
        script = tmp_path / "train.py"
        script.touch()
        assert get_dna_factory_version(str(script)) is None


class TestPrintDnaFactoryBanner:
    """Banner prints the pyproject version, or Unknown as fallback"""

    def test_banner_shows_pyproject_version(self, tmp_path, caplog):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "dummy"\nversion = "9.9.9"\n'
        )
        script = tmp_path / "train.py"
        script.touch()
        logger = logging.getLogger("test_banner_version")
        with caplog.at_level(logging.INFO, logger="test_banner_version"):
            print_dna_factory_banner(logger, str(script))
        assert "🏷️ Version: v9.9.9" in caplog.text

    def test_banner_shows_unknown_without_pyproject(self, tmp_path, caplog):
        script = tmp_path / "train.py"
        script.touch()
        logger = logging.getLogger("test_banner_unknown")
        with caplog.at_level(logging.INFO, logger="test_banner_unknown"):
            print_dna_factory_banner(logger, str(script))
        assert "🏷️ Version: Unknown" in caplog.text
