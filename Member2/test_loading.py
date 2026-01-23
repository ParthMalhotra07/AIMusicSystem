"""
Test script to load features from Member 1
"""

from data_loading import load_features, find_latest_feature_file

# Option 1: Auto-find the latest feature file from Member 1
print("=" * 60)
print("Searching for Member 1's output...")
print("=" * 60)

latest_file = find_latest_feature_file(
    '../Member1/output',
    format='parquet'
)

if latest_file:
    print(f"✓ Found: {latest_file}\n")

    # Load with automatic preprocessing
    dataset, preprocessor = load_features(
        latest_file,
        preprocess=True,
        scaling_method='zscore',
        validate=True,
        verbose=True
    )

    # Print summary
    print("\n")
    dataset.summary()

    # Access the data
    print("\n" + "=" * 60)
    print("Data Access Examples")
    print("=" * 60)

    X = dataset.get_feature_matrix()
    print(f"Feature matrix shape: {X.shape}")
    print(f"First 5 song names: {dataset.get_song_names()[:5]}")

    # Show a single sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Song: {sample['song_name']}")
    print(f"  Features shape: {sample['features'].shape}")
    if 'metadata' in sample:
        print(f"  Tempo: {sample['metadata'].get('tempo_bpm', 'N/A')} BPM")
        print(f"  Duration: {sample['metadata'].get('duration_seconds', 'N/A')} sec")

    print("\n✅ Data loading successful!")
    print(f"Ready for embedding with {len(dataset)} songs × {X.shape[1]} features")

else:
    print("❌ No feature files found in ../Member1/output/")
    print("\nPlease ensure Member 1 has processed audio files first:")
    print("  cd ../Member1")
    print("  python main.py --input_dir ./sample_audio --output_dir ./output")

print("\n" + "=" * 60)
