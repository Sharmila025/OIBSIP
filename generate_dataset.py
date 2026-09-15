"""
Generates a synthetic used-car dataset that mirrors the structure and
statistical patterns of the well-known "Vehicle dataset from CarDekho"
(Kaggle). This environment has no internet access, so the real CSV
cannot be downloaded here -- this script produces a stand-in with the
same columns, realistic brand/model names, and intentionally messy
values (inconsistent casing, whitespace, some duplicates, some nulls)
so the notebook's cleaning steps have real work to do.

If you have downloaded the real Kaggle file (commonly named
`CAR DETAILS FROM CAR DEKHO.csv` or `Car details v3.csv`), just drop it
in the data/ folder as car_data_raw.csv with the same column names and
skip this script -- the rest of the notebook will work unchanged.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 1200

brand_models = {
    "Maruti": ["Swift", "Baleno", "Alto", "Wagon R", "Dzire", "Ertiga", "Vitara Brezza"],
    "Hyundai": ["i20", "Creta", "Venue", "Verna", "Grand i10", "Santro"],
    "Honda": ["City", "Amaze", "Jazz", "WR-V", "Civic"],
    "Toyota": ["Innova", "Fortuner", "Etios", "Glanza", "Yaris"],
    "Ford": ["EcoSport", "Figo", "Endeavour", "Aspire"],
    "Tata": ["Nexon", "Tiago", "Altroz", "Harrier", "Safari"],
    "Mahindra": ["XUV500", "Scorpio", "Bolero", "XUV300", "Thar"],
    "Renault": ["Kwid", "Duster", "Triber"],
    "Volkswagen": ["Polo", "Vento", "Ameo"],
    "Skoda": ["Rapid", "Octavia", "Superb"],
    "BMW": ["3 Series", "5 Series", "X1"],
    "Audi": ["A4", "A6", "Q3"],
}

# Rough base price (in INR lakhs, when brand-new) used to drive the
# synthetic selling-price formula -- premium brands cost more, etc.
brand_base_price = {
    "Maruti": 6.5, "Hyundai": 8.0, "Honda": 9.0, "Toyota": 11.5,
    "Ford": 8.5, "Tata": 8.0, "Mahindra": 10.0, "Renault": 6.5,
    "Volkswagen": 9.5, "Skoda": 12.0, "BMW": 45.0, "Audi": 48.0,
}

fuel_types_clean = ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
# messy variants that a real scraped dataset might contain
fuel_variants = {
    "Petrol": ["Petrol", "petrol", "PETROL", " Petrol"],
    "Diesel": ["Diesel", "diesel", "DIESEL", "Diesel "],
    "CNG": ["CNG", "cng"],
    "LPG": ["LPG", "lpg"],
    "Electric": ["Electric", "electric"],
}

seller_types_clean = ["Individual", "Dealer", "Trustmark Dealer"]
seller_variants = {
    "Individual": ["Individual", "individual", "INDIVIDUAL"],
    "Dealer": ["Dealer", "dealer", "DEALER "],
    "Trustmark Dealer": ["Trustmark Dealer", "trustmark dealer"],
}

transmission_clean = ["Manual", "Automatic"]
transmission_variants = {
    "Manual": ["Manual", "manual", "MANUAL"],
    "Automatic": ["Automatic", "automatic", " Automatic"],
}

owner_types = [
    "First Owner", "Second Owner", "Third Owner",
    "Fourth & Above Owner", "Test Drive Car",
]

rows = []
brands = list(brand_models.keys())
brand_weights = np.array([3, 3, 2, 2, 2, 2.5, 2, 1.5, 1.5, 1, 0.6, 0.6])
brand_weights = brand_weights / brand_weights.sum()

for i in range(N):
    brand = rng.choice(brands, p=brand_weights)
    model = rng.choice(brand_models[brand])
    year = int(rng.integers(2007, 2024))
    age = 2024 - year

    fuel_clean = rng.choice(fuel_types_clean, p=[0.52, 0.38, 0.06, 0.02, 0.02])
    fuel_raw = rng.choice(fuel_variants[fuel_clean])

    seller_clean = rng.choice(seller_types_clean, p=[0.55, 0.35, 0.10])
    seller_raw = rng.choice(seller_variants[seller_clean])

    trans_clean = rng.choice(transmission_clean, p=[0.82, 0.18])
    trans_raw = rng.choice(transmission_variants[trans_clean])

    owner = rng.choice(owner_types, p=[0.55, 0.25, 0.12, 0.05, 0.03])

    km_driven = max(500, int(rng.normal(age * 12000 + 8000, 9000)))

    base = brand_base_price[brand] * 1e5  # convert lakhs to rupees
    depreciation = np.exp(-0.11 * age)
    km_penalty = max(0.55, 1 - km_driven / 350000)
    fuel_bonus = {"Petrol": 1.0, "Diesel": 1.06, "CNG": 0.95, "LPG": 0.9, "Electric": 1.15}[fuel_clean]
    trans_bonus = 1.18 if trans_clean == "Automatic" else 1.0
    owner_penalty = {
        "First Owner": 1.0, "Second Owner": 0.9, "Third Owner": 0.82,
        "Fourth & Above Owner": 0.72, "Test Drive Car": 1.05,
    }[owner]

    noise = rng.normal(1.0, 0.09)
    price = base * depreciation * km_penalty * fuel_bonus * trans_bonus * owner_penalty * noise
    price = max(35000, price)

    name = f"{brand} {model} {year}"

    rows.append({
        "name": name,
        "year": year,
        "selling_price": round(price, -2),
        "km_driven": km_driven,
        "fuel": fuel_raw,
        "seller_type": seller_raw,
        "transmission": trans_raw,
        "owner": owner,
    })

df = pd.DataFrame(rows)

# --- inject messiness on purpose, mirroring real scraped Kaggle data ---

# 1) duplicate ~3% of rows
dupe_idx = rng.choice(df.index, size=int(0.03 * N), replace=False)
df = pd.concat([df, df.loc[dupe_idx]], ignore_index=True)

# 2) null values scattered in a few columns
for col, frac in [("km_driven", 0.02), ("fuel", 0.01), ("selling_price", 0.005)]:
    null_idx = rng.choice(df.index, size=int(frac * len(df)), replace=False)
    df.loc[null_idx, col] = np.nan

# 3) stray whitespace in the name column
ws_idx = rng.choice(df.index, size=int(0.03 * len(df)), replace=False)
df.loc[ws_idx, "name"] = df.loc[ws_idx, "name"].apply(lambda x: f"  {x}  ")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

out_path = "/home/claude/car_price_project/data/car_data_raw.csv"
df.to_csv(out_path, index=False)
print(f"Wrote {len(df)} rows to {out_path}")
print(df.head())
