"""Property-based tests for project structure validation.

**Feature: smart-energy-copilot, Property 1: Project structure consistency**
**Validates: Requirements 3.1**
"""

import os
import pytest
from pathlib import Path
from hypothesis import given, strategies as st


class TestProjectStructureConsistency:
    """Test project structure consistency properties."""
    
    def test_required_directories_exist(self):
        """Test that all required directories exist in the project structure."""
        # **Feature: smart-energy-copilot, Property 1: Project structure consistency**
        # **Validates: Requirements 3.1**
        
        project_root = Path(__file__).parent.parent.parent
        
        required_directories = [
            "src",
            "src/components",
            "src/services", 
            "src/models",
            "src/database",
            "src/config",
            "tests",
            "tests/unit",
            "tests/property",
            "migrations"
        ]
        
        for directory in required_directories:
            dir_path = project_root / directory
            assert dir_path.exists(), f"Required directory {directory} does not exist"
            assert dir_path.is_dir(), f"Path {directory} exists but is not a directory"
    
    def test_required_files_exist(self):
        """Test that all required configuration and setup files exist."""
        # **Feature: smart-energy-copilot, Property 1: Project structure consistency**
        # **Validates: Requirements 3.1**
        
        project_root = Path(__file__).parent.parent.parent
        
        required_files = [
            "requirements.txt",
            "Dockerfile", 
            "docker-compose.yml",
            "alembic.ini",
            "pytest.ini",
            ".gitignore",
            "README.md",
            ".env.example",
            "src/main.py",
            "src/config/settings.py",
            "src/config/logging.py",
            "src/database/connection.py",
            "migrations/env.py",
            "migrations/script.py.mako"
        ]
        
        for file_path in required_files:
            full_path = project_root / file_path
            assert full_path.exists(), f"Required file {file_path} does not exist"
            assert full_path.is_file(), f"Path {file_path} exists but is not a file"
    
    def test_python_packages_have_init_files(self):
        """Test that all Python packages have __init__.py files."""
        # **Feature: smart-energy-copilot, Property 1: Project structure consistency**
        # **Validates: Requirements 3.1**
        
        project_root = Path(__file__).parent.parent.parent
        
        python_packages = [
            "src",
            "src/components",
            "src/services",
            "src/models", 
            "src/database",
            "src/config",
            "tests",
            "tests/unit",
            "tests/property"
        ]
        
        for package in python_packages:
            init_file = project_root / package / "__init__.py"
            assert init_file.exists(), f"Package {package} missing __init__.py file"
            assert init_file.is_file(), f"__init__.py in {package} is not a file"
    
    @given(st.sampled_from([
        "src/components",
        "src/services", 
        "src/models",
        "src/database",
        "src/config"
    ]))
    def test_package_directories_are_importable(self, package_path):
        """Property: For any core package directory, it should be importable as a Python module."""
        # **Feature: smart-energy-copilot, Property 1: Project structure consistency**
        # **Validates: Requirements 3.1**
        
        project_root = Path(__file__).parent.parent.parent
        package_dir = project_root / package_path
        
        # Directory should exist
        assert package_dir.exists(), f"Package directory {package_path} does not exist"
        assert package_dir.is_dir(), f"Package path {package_path} is not a directory"
        
        # Should have __init__.py
        init_file = package_dir / "__init__.py"
        assert init_file.exists(), f"Package {package_path} missing __init__.py"
        
        # __init__.py should be readable
        assert init_file.is_file(), f"__init__.py in {package_path} is not a file"
        
        # Should be able to read the init file (basic syntax check)
        try:
            with open(init_file, 'r', encoding='utf-8') as f:
                content = f.read()
            # Basic validation that it's a valid Python file
            compile(content, str(init_file), 'exec')
        except Exception as e:
            pytest.fail(f"__init__.py in {package_path} is not valid Python: {e}")
    
    def test_docker_configuration_consistency(self):
        """Test that Docker configuration files are consistent with project structure."""
        # **Feature: smart-energy-copilot, Property 1: Project structure consistency**
        # **Validates: Requirements 3.1**
        
        project_root = Path(__file__).parent.parent.parent
        
        # Check Dockerfile exists and references correct paths
        dockerfile = project_root / "Dockerfile"
        assert dockerfile.exists(), "Dockerfile does not exist"
        
        with open(dockerfile, 'r') as f:
            dockerfile_content = f.read()
        
        # Dockerfile should reference the correct source structure
        assert "COPY src/ ./src/" in dockerfile_content, "Dockerfile doesn't copy src/ directory"
        assert "COPY config/ ./config/" in dockerfile_content, "Dockerfile doesn't copy config/ directory"
        assert "COPY migrations/ ./migrations/" in dockerfile_content, "Dockerfile doesn't copy migrations/ directory"
        
        # Check docker-compose.yml
        compose_file = project_root / "docker-compose.yml"
        assert compose_file.exists(), "docker-compose.yml does not exist"
    
    def test_configuration_files_consistency(self):
        """Test that configuration files are consistent with project requirements."""
        # **Feature: smart-energy-copilot, Property 1: Project structure consistency**
        # **Validates: Requirements 3.1**
        
        project_root = Path(__file__).parent.parent.parent
        
        # Check requirements.txt has essential dependencies
        requirements_file = project_root / "requirements.txt"
        assert requirements_file.exists(), "requirements.txt does not exist"
        
        with open(requirements_file, 'r') as f:
            requirements_content = f.read()
        
        essential_deps = [
            "fastapi",
            "sqlalchemy", 
            "alembic",
            "pytest",
            "hypothesis",
            "structlog"
        ]
        
        for dep in essential_deps:
            assert dep in requirements_content, f"Essential dependency {dep} not found in requirements.txt"
        
        # Check pytest.ini configuration
        pytest_ini = project_root / "pytest.ini"
        assert pytest_ini.exists(), "pytest.ini does not exist"
        
        with open(pytest_ini, 'r') as f:
            pytest_content = f.read()
        
        assert "testpaths = tests" in pytest_content, "pytest.ini doesn't specify correct test paths"
        assert "property: Property-based tests" in pytest_content, "pytest.ini missing property test marker"
