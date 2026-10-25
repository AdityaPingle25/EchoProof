from datasets import load_dataset

DATASET_ID = "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA"

print("Connecting to dataset...")

ds = load_dataset(
DATASET_ID,
split="test",
streaming=True
)

print("Dataset features:")
print(ds.features)

print("\nReading one example...")

example = next(iter(ds))

print("Filename:", example["path"])
print("Label:", example["label"])
print("Metadata:", example["notes"])

audio = example["audio"]

print("Audio sample rate:", audio["sampling_rate"])
print("Audio shape:", audio["array"].shape)

print("\nDataset inspection completed.")
