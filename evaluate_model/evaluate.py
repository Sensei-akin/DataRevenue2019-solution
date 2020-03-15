import click
import numpy as np
from pathlib import Path
from scipy import sparse
from sklearn import metrics
import dill


def load_dataset(indir: Path):
    """Get train data set which was stored"""
    X_test = indir / 'Xtest.npz/'
    y_test = indir / 'y_test.npy/'
    Xtest = sparse.load_npz(str(X_test))
    y_test = np.load(str(y_test))

    return Xtest, y_test

@click.command()
@click.option("--in-model-path")
@click.option("--out-dir")
@click.option("--in-data")

def load_model(in_model_path, out_dir, in_data):

    out_dir = Path(out_dir)
    in_model = str(Path(in_model_path )/ 'finalised_model.sav/')
    in_data = Path(in_data)
    out_dir.mkdir(parents=True, exist_ok=True)

    Xtest, y_test = load_dataset(in_data)

    with open(str(in_model), 'rb') as file:
        model = dill.load(file)

    y_pred = model.predict(Xtest)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    

    flag2 = out_dir / '.successEval'
    flag2.touch()

if __name__ == "__main__":
    load_model()

