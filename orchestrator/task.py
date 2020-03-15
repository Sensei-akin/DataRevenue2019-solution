import logging

import luigi
import os
from pathlib import Path

from util import DockerTask

VERSION = os.getenv('PIPELINE_VERSION', '0.1')


class Debug(DockerTask):
    """Use this task with appropriate image to debug things."""

    @property
    def image(self):
        return 'code-challenge/download-data:0.1'

    @property
    def command(self):
        return [
            'sleep', '3600'
        ]


class DownloadData(DockerTask):
    """Initial pipeline task downloads dataset."""

    fname = luigi.Parameter(default='wine_dataset')
    out_dir = luigi.Parameter(default='/usr/share/data/raw/')
    url = luigi.Parameter(
        default='https://github.com/datarevenue-berlin/code-challenge-2019/'
                'releases/download/0.1.0/dataset_sampled.csv'
    )

    @property
    def image(self):
        return f'code-challenge/download-data:{VERSION}'

    @property
    def command(self):
        return [
            'python', 'download_data.py',
            '--name', self.fname,
            '--url', self.url,
            '--out-dir', self.out_dir
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(out_dir/f'{self.fname}.csv')
        )


class MakeDatasets(DockerTask):

    
    in_csv = luigi.Parameter(default='/usr/share/data/raw/wine_dataset.csv')
    out_dir = luigi.Parameter(default='/usr/share/data/processed/')

    @property
    def image(self):
        return f'code-challenge/make-dataset:{VERSION}'

    def requires(self):
        return DownloadData()

    @property
    def command(self):
        # TODO: implement correct command
        # Try to get the input path from self.requires() ;)
        return [
            'python', 'dataset.py',
            '--in-csv', self.in_csv,
            '--out-dir', self.out_dir
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )

class TrainModel(DockerTask):
    """Initial pipeline task train dataset."""

    out_dir = luigi.Parameter('/usr/share/data/processed/model/')
    in_train_data = luigi.Parameter('/usr/share/data/processed/')
    name = luigi.Parameter('finalised_model')

    @property
    def image(self):
        return f'code-challenge/trainmodel:{VERSION}'

    def requires(self):
        return MakeDatasets()
    
    @property
    def command(self):
        # TODO: implement correct command
        # Try to get the input path from self.requires() ;)
        return [
            'python', 'Model.py',
            '--in-train-data', self.in_train_data,
            '--out-dir', self.out_dir,
            '--name', self.name
            ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(Path(self.out_dir)/f'{self.name}.sav')
        )


class EvaluateModel(DockerTask):
    """Initial pipeline task train dataset."""

    out_dir = luigi.Parameter('/usr/share/data/evaluate/')
    in_model_path = luigi.Parameter('/usr/share/data/processed/model/')
    in_data = luigi.Parameter('/usr/share/data/processed/')

    @property
    def image(self):
        return f'code-challenge/evaluate-model:{VERSION}'

    def requires(self):
        return TrainModel()
    
    @property
    def command(self):
        # TODO: implement correct command
        # Try to get the input path from self.requires() ;)
        return [
            'python', 'evaluate.py',
            '--in-data', self.in_data,
            '--in-model-path', self.in_model_path,
            '--out-dir', self.out_dir
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )
