import pandas as pd

# Load metadata
df = pd.read_csv("C:/Users/adity/Documents/Semester 8/Healthcare Analytics Theory/Datasets/HAM10000_metadata.csv")

# Fill missing values
df['age'] = df['age'].fillna(df['age'].median())
df['sex'] = df['sex'].fillna("unknown")
df['localization'] = df['localization'].fillna("unknown")

df = df[['image_id', 'dx', 'age', 'sex', 'localization', 'dx_type']]

# Fixed class order (DO NOT CHANGE)
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Filter valid classes only
df = df[df['dx'].isin(CLASSES)]

# Encode labels
label_map = {cls: idx for idx, cls in enumerate(CLASSES)}
df['label'] = df['dx'].map(label_map)

# Sanity check
print("Samples:", len(df))
print("\nClass distribution:")
print(df['dx'].value_counts())

print("\nLabel map:", label_map)

# Save for model training
df.to_csv("preprocessed_metadata.csv", index=False)
