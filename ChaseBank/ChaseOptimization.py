import pandas as pd
from pulp import *
from haversine import haversine



# Read branch data and keep first 500 rows
df = pd.read_csv("database.csv").iloc[:500].copy()

# Data check
print(df.info())
print(df[["Latitude", "Longitude"]].isna().sum())
print(df[df["Latitude"].isna() | df["Longitude"].isna()])

# Use to know which branch lat/long was filled with avg
df["location_imputed"] = df["Latitude"].isna() | df["Longitude"].isna()


# Fill missing lat/long using County average for preservation
df["Latitude"] = df.groupby("County")["Latitude"].transform(
    lambda x: x.fillna(x.mean())
)
df["Longitude"] = df.groupby("County")["Longitude"].transform(
    lambda x: x.fillna(x.mean())
)
# Data Check
print(df[["Latitude", "Longitude"]].isna().sum())
print(df[df["Latitude"].isna() | df["Longitude"].isna()])

# Use recent deposits to measure branch importance
recent_years = ["2014 Deposits", "2015 Deposits", "2016 Deposits"]
df["recent_avg_deposits"] = df[recent_years].mean(axis=1)

branches = df.index.tolist()

# Find nearby branches
nearest = {}

# Compute distances and keep 10 closest branches for each branch
for i in branches:
    dists = []
    for j in branches:
        d = haversine(
            (df.loc[i, "Latitude"], df.loc[i, "Longitude"]),
            (df.loc[j, "Latitude"], df.loc[j, "Longitude"])
        )
        dists.append((j, d))
    nearest[i] = sorted(dists, key=lambda x: x[1])[:10]

# Number of branches to keep open
K = 200

# Optimization model
# Create minimization problem
model = LpProblem("Branch_Selection", LpMinimize)

# Decision 1: whether each branch is open (1=open, 0=closed)
x = LpVariable.dicts("open", branches, cat="Binary")

# Decision 2: assign each branch to a nearby open branch
y = LpVariable.dicts(
    "assign",
    [(i, j) for i in branches for (j, _) in nearest[i]],
    cat="Binary"
)

# Minimize weighted distance to assigned open branch
model += lpSum(
    df.loc[i, "recent_avg_deposits"] * dist * y[(i, j)]
    for i in branches
    for (j, dist) in nearest[i]
)

# Constraints
# Each branch must be assigned exactly once
for i in branches:
    model += lpSum(y[(i, j)] for (j, _) in nearest[i]) == 1

# Branches can only be assigned to open branches
for (i, j) in y:
    model += y[(i, j)] <= x[j]

# Exactly K/200 branches must be open
model += lpSum(x[j] for j in branches) == K

# Solve

model.solve()
print("Status:", LpStatus[model.status])

# Results
# Mark open vs closed branches
df["Open"] = df.index.map(lambda i: x[i].value() == 1)

# Get assignment for each branch
assignments = {i: j for (i, j) in y if y[(i, j)].value() == 1}
df["Assigned_To"] = df.index.map(lambda i: assignments.get(i, i))

# Count how many branches each open branch serves
df["Branches_Served"] = df.groupby("Assigned_To")["Assigned_To"].transform("count")

# Sum deposits handled by each open branch
df["Deposits_Served"] = df.groupby("Assigned_To")["recent_avg_deposits"].transform("sum")

# Show summary for open branches
open_branches = df[df["Open"]]
for _, row in open_branches.iterrows():
    print(
        f"{row['Branch Name']}: "
        f"Serves {row['Branches_Served']} branches, "
        f"Deposits {row['Deposits_Served']:.2f}"
    )

df.to_csv("branch_consolidation_results.csv", index=False)