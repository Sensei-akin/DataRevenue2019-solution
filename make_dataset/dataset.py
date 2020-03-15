import click
import dask.dataframe as dd
import pandas as pd
import numpy as np
from distributed import Client
from pathlib import Path
import logging
import pandas as pd
from CustomTransfomer import TitleTransformer,DescriptionTransformer 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import fastparquet
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer 
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

def _save_datasets(X_train, X_test, y_train, y_test, outdir: Path):
    """Save data sets into nice directory structure and write SUCCESS flag."""
    print('converting to parquet')

    Xout_train = outdir / 'Xtrain.npz/'
    Xout_test = outdir / 'Xtest.npz/'
    y_out_train = outdir / 'y_train/'
    y_out_test = outdir / 'y_test/'
    flag = outdir / '.SUCCESS'

    sparse.save_npz(str(Xout_train),X_train)
    sparse.save_npz(str(Xout_test),X_test)
    
    np.save(str(y_out_train),y_train)
    np.save(str(y_out_test),y_test)


    flag.touch()

@click.command()
@click.option('--in-csv')
@click.option('--out-dir')
def make_datasets(in_csv, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Connect to the dask cluster
    #c = Client('dask-scheduler:8786')

    # load data as a dask Dataframe if you have trouble with dask
    # please fall back to pandas or numpy
    ddf = pd.read_csv(in_csv)

    # we set the index so we can properly execute loc below
    ddf = ddf.set_index('Unnamed: 0')

    # trigger computation
    n_samples = len(ddf)

    # TODO: implement proper dataset creation here
    # http://docs.dask.org/en/latest/dataframe-api.html

    # split dataset into train test feel free to adjust test percentage
    idx = np.arange(n_samples)
    test_idx = idx[:n_samples // 10]
    test = ddf.loc[test_idx]

    train_idx = idx[n_samples // 10:]
    train = ddf.loc[train_idx]

    numeric_features = ['price']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))])


    categorical_features = ['country', 'winery', 'variety','province']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    description_features = 'description'
    description_transformer = Pipeline(steps=[
        ('tfdif_features', TfidfVectorizer(stop_words='english'))])

    description_length = 'description'
    description_length_transformer = Pipeline(steps=[
        ('length', DescriptionTransformer())])

    age_feature = 'title'
    age_transformer = Pipeline(
        steps=[
        ('age', TitleTransformer()),
        ('imputer', SimpleImputer())
        ])

    preprocessor = ColumnTransformer(
    transformers=[
        ('desc',description_transformer,description_features),
         ('age',age_transformer,age_feature),
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('len',description_length_transformer,description_length)
    ])
         
    # # split dataset into train test feel free to adjust test percentage
    X = preprocessor.fit_transform(ddf)
    y = ddf['points']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

    regr = RandomForestRegressor()
    model = regr.fit(X_train,y_train)
    flag1 = out_dir / '.successTrain'
    flag1.touch()

    y_pred = model.predict(X_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    flag2 = out_dir / '.successEval'
    flag2.touch()

    _save_datasets(X_train, X_test , y_train , y_test , out_dir)
    
if __name__ == '__main__':

    make_datasets()
