import muspy

print("Starting EMOPIA download...")

dataset = muspy.EMOPIADataset(
    root="classifier/data/emopia",
    download_and_extract=True
)

print("EMOPIA downloaded successfully!")
print("Number of samples:", len(dataset))