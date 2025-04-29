import os
import pandas as pd


CSV_PATH = '/home/jovyan/classes_descriptions.csv'
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} classes description")

def get_custom_metadata(info, audio):
    bird_species = os.path.basename(os.path.dirname(info["relpath"]))

    species_samples = df.loc[df['specie'] == bird_species]
    selected_row = species_samples.sample().iloc[0]

    return {"prompt": selected_row['response']}
