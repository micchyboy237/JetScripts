import os
import pytest
from find_large_folders import find_large_folders, format_size, calculate_total_size


class TestFindLargeFolders:
    def test_find_large_folders_depth_zero(self, tmp_path):
        # Setup: Create a temporary directory structure with files
        base_dir = tmp_path / "test_base"
        base_dir.mkdir()
        dir1 = base_dir / "dir1"
        dir1.mkdir()
        dir2 = base_dir / "dir2"
        dir2.mkdir()
        dir1_sub = dir1 / "subdir1"
        dir1_sub.mkdir()

        # Create files to simulate sizes (1 MB = 1_000_000 bytes)
        with open(dir1 / "file1.txt", "wb") as f:
            f.write(b"0" * 1_000_000)  # 1 MB
        with open(dir2 / "file2.txt", "wb") as f:
            f.write(b"0" * 2_000_000)  # 2 MB
        with open(dir1_sub / "file3.txt", "wb") as f:
            f.write(b"0" * 500_000)  # 0.5 MB

        # Expected: Only immediate subdirectories with size >= 1 MB
        expected = [
            {"size": 2.0, "file": str(dir2), "depth": 0},
            {"size": 1.5, "file": str(dir1), "depth": 0},
        ]

        # Result: Collect folders from find_large_folders
        result = []
        for folder_data in find_large_folders(
            base_dir=str(base_dir),
            includes=["*"],
            excludes=[],
            min_size_mb=1,
            max_forward_depth=0
        ):
            result.append(
                {"size": folder_data["size"], "file": folder_data["file"], "depth": folder_data["depth"]})

        # Assert: Compare result with expected
        assert sorted(result, key=lambda x: x["size"], reverse=True) == sorted(
            expected, key=lambda x: x["size"], reverse=True), "Should return only immediate subdirectories with size >= 1 MB at depth 0"

    def test_find_large_folders_depth_one(self, tmp_path):
        # Setup: Create a temporary directory structure with files
        base_dir = tmp_path / "test_base"
        base_dir.mkdir()
        dir1 = base_dir / "dir1"
        dir1.mkdir()
        dir2 = base_dir / "dir2"
        dir2.mkdir()
        dir1_sub = dir1 / "subdir1"
        dir1_sub.mkdir()

        # Create files to simulate sizes
        with open(dir1 / "file1.txt", "wb") as f:
            f.write(b"0" * 1_000_000)  # 1 MB
        with open(dir2 / "file2.txt", "wb") as f:
            f.write(b"0" * 2_000_000)  # 2 MB
        with open(dir1_sub / "file3.txt", "wb") as f:
            f.write(b"0" * 1_500_000)  # 1.5 MB

        # Expected: Directories up to depth 1 with size >= 1 MB
        expected = [
            {"size": 2.5, "file": str(dir1), "depth": 0},
            {"size": 2.0, "file": str(dir2), "depth": 0},
            {"size": 1.5, "file": str(dir1_sub), "depth": 1},
        ]

        # Result: Collect folders from find_large_folders
        result = []
        for folder_data in find_large_folders(
            base_dir=str(base_dir),
            includes=["*"],
            excludes=[],
            min_size_mb=1,
            max_forward_depth=1
        ):
            result.append(
                {"size": folder_data["size"], "file": folder_data["file"], "depth": folder_data["depth"]})

        # Assert: Compare result with expected
        assert sorted(result, key=lambda x: x["size"], reverse=True) == sorted(
            expected, key=lambda x: x["size"], reverse=True), "Should return directories up to depth 1 with size >= 1 MB"

    def test_find_large_folders_depth_zero_no_subdirs(self, tmp_path):
        # Setup: Create a temporary directory structure with files
        base_dir = tmp_path / "test_base"
        base_dir.mkdir()
        dir1 = base_dir / "dir1"
        dir1.mkdir()
        dir2 = base_dir / "dir2"
        dir2.mkdir()

        # Create files to simulate sizes
        with open(dir1 / "file1.txt", "wb") as f:
            f.write(b"0" * 1_000_000)  # 1 MB
        with open(dir2 / "file2.txt", "wb") as f:
            f.write(b"0" * 2_000_000)  # 2 MB

        # Expected: Only immediate subdirectories with size >= 1 MB
        expected = [
            {"size": 2.0, "file": str(dir2), "depth": 0},
            {"size": 1.0, "file": str(dir1), "depth": 0},
        ]

        # Result: Collect folders from find_large_folders
        result = []
        for folder_data in find_large_folders(
            base_dir=str(base_dir),
            includes=["*"],
            excludes=[],
            min_size_mb=1,
            max_forward_depth=0
        ):
            result.append(
                {"size": folder_data["size"], "file": folder_data["file"], "depth": folder_data["depth"]})

        # Assert: Compare result with expected
        assert sorted(result, key=lambda x: x["size"], reverse=True) == sorted(
            expected, key=lambda x: x["size"], reverse=True), "Should return only immediate subdirectories with size >= 1 MB, no deeper subdirs"

    def test_find_large_folders_below_min_size(self, tmp_path):
        # Setup: Create a temporary directory structure with small files
        base_dir = tmp_path / "test_base"
        base_dir.mkdir()
        dir1 = base_dir / "dir1"
        dir1.mkdir()
        with open(dir1 / "file1.txt", "wb") as f:
            f.write(b"0" * 500_000)  # 0.5 MB

        # Expected: No folders since size < 1 MB
        expected = []

        # Result: Collect folders from find_large_folders
        result = []
        for folder_data in find_large_folders(
            base_dir=str(base_dir),
            includes=["*"],
            excludes=[],
            min_size_mb=1,
            max_forward_depth=0
        ):
            result.append(
                {"size": folder_data["size"], "file": folder_data["file"], "depth": folder_data["depth"]})

        # Assert: Compare result with expected
        assert result == expected, "Should return no folders when all are below min_size_mb"

    def test_format_size(self):
        # Test cases for format_size
        test_cases = [
            (500, "500.00 MB"),
            (1000, "1.00 GB"),
            (1500, "1.50 GB"),
        ]
        for size_mb, expected in test_cases:
            result = format_size(size_mb)
            assert result == expected, f"Expected {expected} for {size_mb} MB, got {result}"

    def test_calculate_total_size(self):
        # Test cases for calculate_total_size
        folders = [
            {"size": 1000, "file": "dir1", "depth": 0},
            {"size": 500, "file": "dir1/subdir1", "depth": 1},
            {"size": 2000, "file": "dir2", "depth": 0},
        ]
        expected = 3000.0  # Only depth 0 folders
        result = calculate_total_size(folders)
        assert result == expected, f"Expected total size {expected}, got {result}"
