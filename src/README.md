Gowalla `src` package

This package contains modules for data loading, preprocessing, candidate generation,
feature computation, modeling, and evaluation for the link-prediction experiments.

Files and responsibilities:

- `load_data.py`: read dataset files into pandas DataFrames.
- `preprocess.py`: cleaning and normalization utilities.
- `candidate_generation.py`: produce candidate node pairs to score.
- `topology_features.py`: compute graph-based features like common neighbors and Jaccard.
- `geographic_features.py`: spatial distance and related features.
- `entropy_features.py`: user/location entropy statistics.
- `models.py`: estimator wrappers and training helpers.
- `evaluation.py`: metrics like AUC and Average Precision.

Usage

From the project root run small smoke test in Python:

```bash
python -c "from src import load_data; print('ok')"
```

Proceed by filling implementation details as needed.
