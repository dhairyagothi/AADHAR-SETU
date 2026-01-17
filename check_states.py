import pandas as pd
from pathlib import Path

# Define paths
biometric_path = r"c:\personal dg\github_repo\aadhar-dataset\api_data_aadhar_biometric\api_data_aadhar_biometric"
demographic_path = r"c:\personal dg\github_repo\aadhar-dataset\api_data_aadhar_demographic\api_data_aadhar_demographic"
enrolment_path = r"c:\personal dg\github_repo\aadhar-dataset\api_data_aadhar_enrolment"

print("="*80)
print("INVESTIGATING STATE NAME VARIATIONS")
print("="*80)

# Check each dataset
datasets = {
    "BIOMETRIC": biometric_path,
    "DEMOGRAPHIC": demographic_path,
    "ENROLMENT": enrolment_path
}

all_states = {}

for dataset_name, folder_path in datasets.items():
    print(f"\n{dataset_name} DATASET - Unique States:")
    print("-" * 80)
    
    csv_files = list(Path(folder_path).glob("*.csv"))
    
    states_set = set()
    for file in sorted(csv_files):
        try:
            df = pd.read_csv(file)
            unique_states = df['state'].unique().tolist()
            states_set.update(unique_states)
        except Exception as e:
            print(f"Error reading {file.name}: {e}")
    
    states_list = sorted(list(states_set))
    all_states[dataset_name] = states_list
    
    print(f"Total count: {len(states_list)}\n")
    for i, state in enumerate(states_list, 1):
        print(f"{i:2d}. {state}")

# Find differences
print("\n" + "="*80)
print("COMPARATIVE ANALYSIS")
print("="*80)

bio_states = set(all_states["BIOMETRIC"])
dem_states = set(all_states["DEMOGRAPHIC"])
enr_states = set(all_states["ENROLMENT"])

print(f"\nStates in BIOMETRIC only: {bio_states - dem_states - enr_states}")
print(f"States in DEMOGRAPHIC only: {dem_states - bio_states - enr_states}")
print(f"States in ENROLMENT only: {enr_states - bio_states - dem_states}")

# Union of all
all_unique_states = bio_states | dem_states | enr_states
print(f"\nTotal unique states across ALL datasets: {len(all_unique_states)}")
print(f"States: {sorted(all_unique_states)}")

# Check for potential duplicates/variations
print("\n" + "="*80)
print("CHECKING FOR SPELLING VARIATIONS AND CASE ISSUES")
print("="*80)

for state in sorted(all_unique_states):
    # Check for case variations
    lowercase = state.lower()
    print(f"{state}")
