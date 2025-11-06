# Environment Setup

This directory contains a complete Python environment for data science and machine learning projects.

## Environment Details

- **Python Version**: 3.11.2
- **Virtual Environment**: `fresh_venv`
- **Environment Location**: `c:\Users\1.casestudy\fresh_venv`

## Activation

### Option 1: PowerShell (VS Code Default)
```powershell
.\fresh_venv\Scripts\Activate.ps1
```

### Option 2: Command Prompt (CMD)
```cmd
fresh_venv\Scripts\activate.bat
```

### Option 3: VS Code Integrated Terminal
1. Open VS Code terminal (`Ctrl + ` ` or View → Terminal)
2. Navigate to your project directory:
   ```cmd
   cd "c:\Users\1.casestudy"
   ```
3. Activate using CMD:
   ```cmd
   fresh_venv\Scripts\activate.bat
   ```
4. Or activate using PowerShell:
   ```powershell
   .\fresh_venv\Scripts\Activate.ps1
   ```

### Option 4: VS Code Python Interpreter Selection
1. Open VS Code in your project folder
2. Press `Ctrl + Shift + P` to open Command Palette
3. Type "Python: Select Interpreter"
4. Choose the interpreter from: `.\fresh_venv\Scripts\python.exe`

**Note**: When activated, you'll see `(fresh_venv)` at the beginning of your command prompt.

## Installed Packages

The following packages are installed and verified:

### Core Data Science Libraries
- **numpy**: 1.26.4 (compatible with pmdarima)
- **pandas**: 2.3.3
- **matplotlib**: 3.10.7
- **seaborn**: 0.13.2
- **scikit-learn**: 1.7.2
- **scipy**: 1.16.3

### Machine Learning & Deep Learning
- **tensorflow**: 2.20.0
- **keras**: 3.12.0
- **xgboost**: 3.1.1

### Time Series Analysis
- **statsmodels**: 0.14.5
- **pmdarima**: 2.0.4
- **prophet**: 1.2.1

### Specialized Libraries
- **linear-tree**: 0.3.5 (LinearBoostRegressor)
- **holidays**: 0.83
- **workalendar**: 17.0.0
- **joblib**: 1.5.2
- **requests**: 2.32.5

## Files

- `requirements.txt`: Complete list of installed packages with versions
- `test_imports.py`: Test script to verify all imports work correctly

## Usage

1. Activate the virtual environment
2. Run `python test_imports.py` to verify all packages are working
3. Start your data science project!

## Notes

- The numpy version is specifically set to < 2.0 to maintain compatibility with pmdarima
- TensorFlow may show informational messages about oneDNN optimizations (these are normal)
- Prophet may show a warning about plotly not being installed (optional for basic usage)

## Import Template

```python
import datetime
import json
import os
from joblib import Parallel, delayed
from time import sleep, time
import logging

import itertools
import holidays
import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
import requests
import seaborn as sns
import statsmodels
import statsmodels.tsa.api as sm
import tensorflow as tf
import xgboost as xgb
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
from lineartree import LinearBoostRegressor
from matplotlib import rcParams  # Used to set default parameters
from prophet import Prophet
from prophet.diagnostics import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import month_plot, plot_acf, plot_pacf, quarter_plot
from workalendar.europe import UnitedKingdom
```
