import dill
import click
import numpy as np
from pathlib import Path
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib




def load_dataset(indir: Path):
    """Get train data set which was stored"""
    X_train = indir / 'Xtrain.npz/'
    y_train = indir / 'y_train.npy/'
    Xtrain = sparse.load_npz(str(X_train))
    y_train = np.load(str(y_train))

    return Xtrain, y_train

def _save_model(model,out_dir, name):
    """save your trained model """

    flag = out_dir/'.successModel'
    model_dir = out_dir/f'{name}.sav'
    dill.dump(model, open(model_dir, 'wb'))

    flag.touch()

@click.command()
@click.option('--in-train-data')
@click.option('--out-dir')
@click.option('--name')

def build_model(in_train_data, out_dir, name):
    out_dir = Path(out_dir)
    in_train_data = Path(in_train_data)
    #out_dir.mkdir(parents=True, exist_ok=True)

    Xtrain, X_test = load_dataset(in_train_data)
    
    regr = RandomForestRegressor()
    model = regr.fit(Xtrain,X_test)
    _save_model(model,out_dir,name)


if __name__ == "__main__":
    build_model()