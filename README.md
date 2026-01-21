# ChaseBankProject
This repository contains two Python projects/notebooks that optimize bank branch consolidation using mathematical optimization and analyze the resulting branch network using geographic clustering and deposit behavior.

### Project 1: Chase Bank Optimization (PuLP + Haversine)
This project formulates a binary optimization problem to select a fixed number of bank branches to remain open while assigning closed branches to nearby open ones in a way that minimizes customer impact.

**Key objectives:**
- Keep exactly 200 branches open
- Assign every branch to a nearby open branch
- Minimize **distance-weighted deposit impact**
- Preserve geographic coverage using latitude/longitude data

**Techniques used:**
- Integer Linear Programming (PuLP)
- Haversine distance for geospatial proximity
- Deposit weighted optimization
- Data imputation using county level averages

**Key outputs:**
- Open vs closed branch decisions
- Branch-to-branch assignment mapping
- Number of branches served per open branch
- Total deposits handled per open branch

-

### Project 2: Chase Bank Clustering Analysis (KMeans)
This project analyzes the optimized branch network from Project 1 using clustering techniques to identify patterns among surviving branches.

**Key objectives:**
- Cluster branches based on:
  - Geographic location (latitude & longitude)
  - Recent average deposits
- Understand operational and financial differences between clusters

**Techniques used:**
- Feature scaling (StandardScaler)
- KMeans clustering
- Elbow method to determine optimal cluster count
- Cluster-level statistical summaries

**Key insights produced:**
- Cluster sizes and geographic distribution
- Average and total deposits per cluster
- Branch load (branches served) by cluster
- County and state-level breakdowns
