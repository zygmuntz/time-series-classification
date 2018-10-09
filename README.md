# Time Series Classification

Install tsfresh (`pip install tsfresh`). 

Edit `config.py` for the dataset you want to handle. By default, it's _Wafer_. Two more are provided in the `data\` directory: _Ford A_ and _Ford B_. You can use any dataset from the [UEA & UCR Time Series Classification Repository](http://www.timeseriesclassification.com/dataset.php).

When ready, run

1. `extract_features.py`
2. `select_features.py`
3. `train_and_evaluate.py`

Step one takes some time, so you can skip it - each dataset directory already contains extracted features.

The code uses Python 2 (print statements).
