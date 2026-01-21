#!/usr/bin/env python3
"""
Audio Feature Engineering - Command Line Interface

Main entry point for extracting audio features from files or directories.

Usage:
    # Process a single file
    python main.py --input song.mp3 --output features.csv
    
    # Process a directory
    python main.py --input_dir ./music --output_dir ./features --format parquet
    
    # With custom config
    python main.py --input_dir ./music --sample_rate 44100 --n_mfcc 20
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, AudioConfig, FeatureConfig, ExportConfig
from pipeline.extractor import AudioFeatureExtractor, extract_features_from_directory
from pipeline.scaler import create_scaler, ZScoreScaler
from export.exporter import (
    export_to_csv,
    export_to_npy,
    export_to_parquet,
    generate_feature_manifest
)
from export.feature_schema import get_feature_names, FEATURE_DEFINITIONS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract audio features for music recommendation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract features from a single file
    python main.py --input song.mp3 --output features.csv
    
    # Extract features from a directory
    python main.py --input_dir ./music --output_dir ./output --format parquet
    
    # Use custom sample rate
    python main.py --input_dir ./music --sample_rate 44100
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input', '-i',
        type=str,
        help='Path to a single audio file'
    )
    input_group.add_argument(
        '--input_dir', '-d',
        type=str,
        help='Path to directory containing audio files'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path (for single file input)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='Output directory (for directory input)'
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['csv', 'npy', 'parquet'],
        default='parquet',
        help='Output format (default: parquet)'
    )
    
    # Audio processing options
    parser.add_argument(
        '--sample_rate', '-sr',
        type=int,
        default=22050,
        help='Target sample rate in Hz (default: 22050)'
    )
    parser.add_argument(
        '--n_mfcc',
        type=int,
        default=13,
        help='Number of MFCCs to extract (default: 13)'
    )
    parser.add_argument(
        '--n_mels',
        type=int,
        default=128,
        help='Number of Mel bands (default: 128)'
    )
    
    # Processing options
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Search subdirectories recursively'
    )
    parser.add_argument(
        '--scale',
        type=str,
        choices=['none', 'minmax', 'zscore'],
        default='zscore',
        help='Feature scaling method (default: zscore)'
    )
    parser.add_argument(
        '--no_metadata',
        action='store_true',
        help='Skip metadata extraction (faster)'
    )
    
    # Verbosity
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    return parser.parse_args()


def create_config(args) -> Config:
    """Create configuration from command line arguments."""
    audio_config = AudioConfig(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
    )
    
    feature_config = FeatureConfig(
        n_mfcc=args.n_mfcc,
    )
    
    export_config = ExportConfig(
        default_format=args.format,
    )
    
    return Config(
        audio=audio_config,
        features=feature_config,
        export=export_config,
    )


def process_single_file(args, config: Config) -> None:
    """Process a single audio file."""
    print(f"\n🎵 Processing: {args.input}")
    print("-" * 50)
    
    extractor = AudioFeatureExtractor(config)
    
    # Extract features with metadata
    features, metadata = extractor.process_file(
        args.input,
        return_metadata=True
    )
    
    print(f"✓ Duration: {metadata['duration_seconds']:.2f}s")
    print(f"✓ Tempo: {metadata['tempo_bpm']:.1f} BPM")
    print(f"✓ Key: {metadata['key']}")
    print(f"✓ Feature vector: {len(features)} dimensions")
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.input)[0]
        output_path = f"{base_name}_features.{args.format}"
    
    # Export
    feature_names = extractor.feature_names
    
    if args.format == 'csv':
        export_to_csv(
            features.reshape(1, -1),
            output_path,
            feature_names=feature_names,
            file_paths=[args.input],
            metadata=[metadata]
        )
    elif args.format == 'npy':
        export_to_npy(
            features.reshape(1, -1),
            output_path,
            feature_names=feature_names,
            file_paths=[args.input]
        )
    elif args.format == 'parquet':
        export_to_parquet(
            features.reshape(1, -1),
            output_path,
            feature_names=feature_names,
            file_paths=[args.input],
            metadata=[metadata]
        )
    
    print(f"\n✅ Saved to: {output_path}")


def process_directory(args, config: Config) -> None:
    """Process a directory of audio files."""
    print(f"\n🎵 Processing directory: {args.input_dir}")
    print("-" * 50)
    
    # Find all audio files
    supported = config.pipeline.supported_formats
    audio_files = []
    
    if args.recursive:
        for root, dirs, files in os.walk(args.input_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in supported:
                    audio_files.append(os.path.join(root, f))
    else:
        for f in os.listdir(args.input_dir):
            if os.path.splitext(f)[1].lower() in supported:
                audio_files.append(os.path.join(args.input_dir, f))
    
    if not audio_files:
        print(f"❌ No audio files found in {args.input_dir}")
        print(f"   Supported formats: {supported}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    # Create extractor
    extractor = AudioFeatureExtractor(config)
    
    # Process all files
    features, metadata_list = extractor.process_batch(
        audio_files,
        show_progress=not args.quiet,
        return_metadata=not args.no_metadata
    )
    
    print(f"\n✓ Processed {len(audio_files)} files")
    print(f"✓ Feature matrix shape: {features.shape}")
    
    # Apply scaling
    if args.scale != 'none':
        print(f"\nApplying {args.scale} scaling...")
        scaler = create_scaler(args.scale)
        features = scaler.fit_transform(features)
        
        # Save scaler
        scaler_path = os.path.join(args.output_dir, f'scaler_{args.scale}.json')
        scaler.save(scaler_path)
        print(f"✓ Scaler saved to: {scaler_path}")
    else:
        scaler_path = None
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'audio_features_{timestamp}.{args.format}'
    output_path = os.path.join(args.output_dir, output_file)
    
    # Export
    feature_names = extractor.feature_names
    
    if args.format == 'csv':
        export_to_csv(
            features,
            output_path,
            feature_names=feature_names,
            file_paths=audio_files,
            metadata=metadata_list if not args.no_metadata else None
        )
    elif args.format == 'npy':
        export_to_npy(
            features,
            output_path,
            feature_names=feature_names,
            file_paths=audio_files
        )
    elif args.format == 'parquet':
        export_to_parquet(
            features,
            output_path,
            feature_names=feature_names,
            file_paths=audio_files,
            metadata=metadata_list if not args.no_metadata else None
        )
    
    print(f"\n✅ Features saved to: {output_path}")
    
    # Generate manifest
    from config import FEATURE_SCHEMA
    manifest_path = generate_feature_manifest(
        args.output_dir,
        feature_names,
        FEATURE_SCHEMA,
        scaler_path
    )
    print(f"✅ Manifest saved to: {manifest_path}")
    
    # Print summary statistics
    print("\n📊 Feature Statistics:")
    print(f"   Min: {features.min():.4f}")
    print(f"   Max: {features.max():.4f}")
    print(f"   Mean: {features.mean():.4f}")
    print(f"   Std: {features.std():.4f}")


def main():
    """Main entry point."""
    args = parse_args()
    config = create_config(args)
    
    print("=" * 60)
    print("🎼 Audio Feature Engineering Pipeline")
    print("=" * 60)
    print(f"Sample Rate: {config.audio.sample_rate} Hz")
    print(f"MFCCs: {config.features.n_mfcc}")
    print(f"Mel Bands: {config.audio.n_mels}")
    print(f"Output Format: {args.format}")
    
    try:
        if args.input:
            process_single_file(args, config)
        else:
            process_directory(args, config)
        
        print("\n✅ Done!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
