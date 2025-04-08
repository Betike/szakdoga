# Sports Match Analysis using PyTorch

This project focuses on analyzing sports matches using various data mining methods and deep learning techniques implemented with PyTorch.

## Project Overview

The project aims to:
- Analyze sports match data using machine learning techniques
- Predict match outcomes
- Analyze player performance
- Identify strategic patterns
- Compare different data mining approaches

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `src/`: Source code directory
  - `data/`: Data loading and preprocessing modules
  - `models/`: PyTorch model implementations
  - `utils/`: Utility functions
  - `visualization/`: Data visualization modules
- `notebooks/`: Jupyter notebooks for analysis and experimentation
- `database.sqlite`: Main database file containing match data

## Usage

1. Data Exploration:
```bash
python src/data/explore_data.py
```

2. Model Training:
```bash
python src/models/train.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 