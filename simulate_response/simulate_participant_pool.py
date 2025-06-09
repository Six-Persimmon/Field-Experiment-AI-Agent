#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File    :   simulate_participant_pool.py
@Time    :   2025/05/15 15:50:53
@Author  :   Manlu Ouyang
@Version :   1.0
@Contact :   mo2615@stern.nyu.edu
@Desc    :   None
'''

import pandas as pd
import random
import uuid
import os

# === Step 1: Define allowed race categories ===
allowed_races = {
    "White": "White",
    "Black or African American": "Black",
    "Asian and Native Hawaiian and Other Pacific Islander": "Asian",
    "American Indian and Alaska Native": "Indigenous",
    "Hispanic or Latino": "Latino"
}

allowed_genders = ["Male", "Female"]

# === Step 2: Get yes/no for race distribution ===
print("Do you want a balanced race distribution among the five races?")
for k, v in allowed_races.items():
    print(f"  {k} → {v}")

while True:
    use_balanced = input("Enter 'yes' for balanced or 'no' to specify custom percentages: ").strip().lower()
    if use_balanced in ['yes', 'no']:
        break
    print("Please enter 'yes' or 'no'.")

# === Step 3: Get race percentages ===
if use_balanced == 'yes':
    race_dist = {k: 1.0 / len(allowed_races) for k in allowed_races}
else:
    print("\n📊 Please enter race percentages (0 - 100).")
    race_raw = {}
    for label in allowed_races:
        while True:
            try:
                percent = float(input(f"How much percentage do you want for {allowed_races[label]}? "))
                if percent < 0:
                    raise ValueError
                race_raw[label] = percent
                break
            except ValueError:
                print("Please enter a valid non-negative number.")
    total = sum(race_raw.values())
    if abs(total - 100) > 1e-2:
        print("Percentages do not sum to 100. Normalizing automatically...")
    race_dist = {k: v / total for k, v in race_raw.items()}

# === Step 4: Get gender percentages ===
print("\n Please enter gender percentages (0 - 100).")
gender_raw = {}
for gender in allowed_genders:
    while True:
        try:
            percent = float(input(f"How much percentage do you want for {gender}? "))
            if percent < 0:
                raise ValueError
            gender_raw[gender] = percent
            break
        except ValueError:
            print("Please enter a valid non-negative number.")
total = sum(gender_raw.values())
if abs(total - 100) > 1e-2:
    print("Percentages do not sum to 100. Normalizing automatically...")
gender_dist = {k: v / total for k, v in gender_raw.items()}

# === Step 5: Age range ===
while True:
    try:
        age_min = int(input("\nEnter minimum age (e.g., 18): "))
        age_max = int(input("Enter maximum age (e.g., 65): "))
        if age_min >= age_max:
            print("Minimum age must be less than maximum age.")
            continue
        break
    except ValueError:
        print("Please enter valid integers.")

# === Step 6: Number of participants ===
while True:
    try:
        num_participants = int(input("\nEnter number of participants to generate: "))
        if num_participants <= 0:
            print("Number must be positive.")
            continue
        break
    except ValueError:
        print("Please enter a valid integer.")

# === Step 7: Generate participants ===
participants = []
for _ in range(num_participants):
    participant_id = str(uuid.uuid4())
    race_key = random.choices(list(race_dist.keys()), weights=race_dist.values())[0]
    race = allowed_races[race_key]
    gender = random.choices(list(gender_dist.keys()), weights=gender_dist.values())[0]
    age = random.randint(age_min, age_max)

    participants.append({
        'ParticipantID': participant_id,
        'Race': race,
        'Gender': gender,
        'Age': age
    })

# === Step 8: Save to CSV ===
df = pd.DataFrame(participants)

# Get directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create full path for CSV file
output_path = os.path.join(script_dir, "participant_pool.csv")

# Save the file
df.to_csv(output_path, index=False)

print(f"\n✅ Participant pool saved to: {output_path}")
