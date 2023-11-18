"""This file contains the dataclasses for each step"""
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    """
    This class will help in reading configurations from config.yaml file 
    for data ingestion step.
    """
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path
