# Big Data Analytics OULAD

Repository for the V4 OULAD project. The planned work has two implementation branches:

- Supervised learning with Logistic Regression and Random Forest to predict course non-completion using information available at the start of day 28.
- Unsupervised learning with K-Means to cluster learner behaviour.
- Pandas/scikit-learn and PySpark DataFrame/MLlib implementations share the `feature_28d` definition and split manifest.

Development and execution are planned for local Ubuntu with Conda and JupyterLab.

## Repository layout

The repository is currently at the setup stage. Add directories when their first maintained file is ready; do not add empty placeholder code files.

| Path | Planned contents |
| --- | --- |
| `config/` | Agreed modeling rules and `feature_28d` schema |
| `data/raw/` | Local source data; excluded from Git |
| `data/processed/` | Locally generated feature data and split manifest, subject to data review |
| `notebooks/pandas/` | Pandas and scikit-learn analysis |
| `notebooks/pyspark/` | PySpark DataFrame and MLlib analysis |
| `src/data/` | Reusable ETL and feature preparation code |
| `src/modeling/` | Supervised model code |
| `src/clustering/` | K-Means code |
| `results/models/`, `results/clustering/`, `results/benchmark/` | Generated models, clustering output, and benchmark output |
| `reports/` | Project and weekly reports |
| `tests/` | Data and modeling checks, when implemented |

## Local data and outputs

Keep OULAD source files and personal Google Form responses on the local machine. Git ignores these data locations, local environments, caches, and generated model files. Scripts, notebooks, configuration, metrics, and reports can be versioned.

The agreed week 1 documents have not yet been added. Place the modeling rules and feature schema under `config/`, and the week 1 report under `reports/` when those source files are available. Their contents must remain the authoritative agreed definitions.
