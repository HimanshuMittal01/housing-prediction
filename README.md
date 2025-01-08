# Chicago Housing Prediction

The objective of this assignment is to predict housing prices in Chicago using tree-based machine learning models. The dataset, scraped from Zillow, contains various property features. The analysis aims to clean, explore, and model the data to generate actionable insights. 

The model achieved 0.842 R<sup>2</sup> score on the OOF predictions in K=5 Fold cross validation. More details in [Project Report](https://docs.google.com/document/d/1NbbfePTH5WJRvjWV-u8L2qZP-bhk7Cda3dx9nMFsPN8/edit?usp=sharing)

**Setup environment:**
```bash
uv venv --python 3.10
source .venv/bin/activate
uv sync
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── raw            <- The original, immutable data dump.
│   └── processed      <- The final, canonical data sets for modeling.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
|
├── notebooks          <- Jupyter notebooks.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         housing_prediction and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── housing_prediction   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes housing_prediction a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Preprocess data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── models.py               <- Build and serialize models
    │
    ├── predict.py              <- Code to run model inference with trained models
    │
    ├── train.py                <- Code to train models
    │
    └── utils.py                <- Code to create visualizations
```

--------

