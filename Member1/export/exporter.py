"""
Data Export Module

Exports feature vectors in multiple formats for different use cases:
- CSV: Human-readable, easy to inspect in Excel/spreadsheets
- NPY: Fast NumPy binary format
- Parquet: Columnar storage for efficient querying and compression
"""

import os
import json
from typing import List, Optional, Dict
from datetime import datetime
import numpy as np
import pandas as pd


def export_to_csv(
    features: np.ndarray,
    output_path: str,
    feature_names: Optional[List[str]] = None,
    file_paths: Optional[List[str]] = None,
    metadata: Optional[List[Dict]] = None,
    float_precision: int = 6
) -> str:
    """
    Export features to CSV format.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    output_path : str
        Output file path
    feature_names : list, optional
        Column names for features
    file_paths : list, optional
        Source file paths to include as column
    metadata : list, optional
        Additional metadata for each sample
    float_precision : int, optional
        Decimal places for floats (default: 6)
        
    Returns
    -------
    str
        Path to saved file
    """
    # Create DataFrame
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(features.shape[1])]
    
    df = pd.DataFrame(features, columns=feature_names)
    
    # Add file paths if provided
    if file_paths is not None:
        df.insert(0, 'file_path', file_paths)
        df.insert(1, 'file_name', [os.path.basename(p) for p in file_paths])
    
    # Add metadata if provided
    if metadata is not None:
        for key in metadata[0].keys():
            if key not in df.columns:
                df[key] = [m.get(key) for m in metadata]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Save
    df.to_csv(output_path, index=False, float_format=f'%.{float_precision}f')
    
    return output_path


def export_to_npy(
    features: np.ndarray,
    output_path: str,
    save_metadata: bool = True,
    file_paths: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None
) -> str:
    """
    Export features to NumPy binary format.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    output_path : str
        Output file path
    save_metadata : bool, optional
        Save metadata JSON alongside (default: True)
    file_paths : list, optional
        Source file paths for metadata
    feature_names : list, optional
        Feature names for metadata
        
    Returns
    -------
    str
        Path to saved file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Save features
    np.save(output_path, features.astype(np.float32))
    
    # Save metadata
    if save_metadata:
        metadata_path = output_path.replace('.npy', '_metadata.json')
        metadata = {
            'shape': list(features.shape),
            'dtype': str(features.dtype),
            'n_samples': features.shape[0],
            'n_features': features.shape[1],
            'created_at': datetime.now().isoformat(),
        }
        
        if file_paths is not None:
            metadata['file_paths'] = file_paths
        
        if feature_names is not None:
            metadata['feature_names'] = feature_names
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return output_path


def export_to_parquet(
    features: np.ndarray,
    output_path: str,
    feature_names: Optional[List[str]] = None,
    file_paths: Optional[List[str]] = None,
    metadata: Optional[List[Dict]] = None,
    compression: str = 'snappy'
) -> str:
    """
    Export features to Parquet format.
    
    Parquet is ideal for:
    - Large datasets (efficient compression)
    - Columnar queries (selecting specific features)
    - Integration with Spark/Dask for distributed processing
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    output_path : str
        Output file path
    feature_names : list, optional
        Column names for features
    file_paths : list, optional
        Source file paths to include as column
    metadata : list, optional
        Additional metadata for each sample
    compression : str, optional
        Compression codec: 'snappy', 'gzip', 'brotli', 'zstd', None
        
    Returns
    -------
    str
        Path to saved file
    """
    # Create DataFrame
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(features.shape[1])]
    
    df = pd.DataFrame(features.astype(np.float32), columns=feature_names)
    
    # Add file paths if provided
    if file_paths is not None:
        df.insert(0, 'file_path', file_paths)
        df.insert(1, 'file_name', [os.path.basename(p) for p in file_paths])
    
    # Add metadata if provided
    if metadata is not None:
        for key in metadata[0].keys():
            if key not in df.columns:
                df[key] = [m.get(key) for m in metadata]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Save
    df.to_parquet(output_path, compression=compression, index=False)
    
    return output_path


def generate_feature_manifest(
    output_dir: str,
    feature_names: List[str],
    schema: Dict,
    scaler_path: Optional[str] = None
) -> str:
    """
    Generate a manifest file documenting the feature vector.
    
    This is crucial for Member 2 and Member 3 to understand the data.
    
    Parameters
    ----------
    output_dir : str
        Directory to save manifest
    feature_names : list
        List of all feature names
    schema : dict
        Feature schema with dimensions and descriptions
    scaler_path : str, optional
        Path to the fitted scaler
        
    Returns
    -------
    str
        Path to manifest file
    """
    manifest = {
        'version': '1.0.0',
        'created_at': datetime.now().isoformat(),
        'total_features': len(feature_names),
        'feature_names': feature_names,
        'schema': schema,
        'scaler_path': scaler_path,
        'description': 'Audio feature vectors for music similarity and recommendation',
        'usage': {
            'loading_csv': "pd.read_csv('features.csv')",
            'loading_npy': "np.load('features.npy')",
            'loading_parquet': "pd.read_parquet('features.parquet')",
        },
        'categories': {
            'timbral': 'Sound texture features (MFCCs, spectral characteristics)',
            'rhythmic': 'Tempo, beat, and onset features',
            'harmonic': 'Pitch and chord progression features',
        }
    }
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    manifest_path = os.path.join(output_dir, 'feature_manifest.json')
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Also generate markdown documentation
    md_path = os.path.join(output_dir, 'FEATURES.md')
    with open(md_path, 'w') as f:
        f.write("# Audio Feature Documentation\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Features:** {len(feature_names)}\n\n")
        
        f.write("## Feature Categories\n\n")
        for category, desc in manifest['categories'].items():
            f.write(f"### {category.title()}\n")
            f.write(f"{desc}\n\n")
        
        f.write("## Feature List\n\n")
        f.write("| Index | Feature Name | Category |\n")
        f.write("|-------|-------------|----------|\n")
        
        for i, name in enumerate(feature_names):
            category = 'timbral' if any(x in name for x in ['mfcc', 'spectral', 'zero_crossing', 'rms']) else \
                       'rhythmic' if any(x in name for x in ['tempo', 'beat', 'onset']) else \
                       'harmonic'
            f.write(f"| {i} | {name} | {category} |\n")
    
    return manifest_path


def load_features(path: str) -> np.ndarray:
    """
    Load features from any supported format.
    
    Parameters
    ----------
    path : str
        Path to feature file (.csv, .npy, or .parquet)
        
    Returns
    -------
    np.ndarray
        Feature matrix
    """
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.npy':
        return np.load(path)
    elif ext == '.csv':
        df = pd.read_csv(path)
        # Drop non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[numeric_cols].values
    elif ext == '.parquet':
        df = pd.read_parquet(path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[numeric_cols].values
    else:
        raise ValueError(f"Unsupported format: {ext}")
