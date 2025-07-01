import logging
from pathlib import Path

from icu_benchmarks.data.split_process_data import preprocess_data
from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.preprocessor import PolarsRegressionPreprocessor

from sepsis_osc.utils.logger import setup_logging
from sepsis_osc.utils.config import yaib_data_dir
from sepsis_osc.model.gin_configs import file_names, vars, modality_mapping
import os

logger = logging.getLogger(__name__)

data_dir = Path(yaib_data_dir)


data = preprocess_data(
    data_dir=data_dir,
    file_names=file_names,
    preprocessor=PolarsRegressionPreprocessor,
    seed=666,
    debug=False,
    generate_cache=True,
    load_cache=True,
    cv_repetitions=5,
    repetition_index=0,
    train_size=None,
    cv_folds=5,
    fold_index=0,
    pretrained_imputation_model=None,
    runmode=RunMode.regression,
    vars=vars,
    modality_mapping=modality_mapping,
)

for si, s in data.items():
    for k, df in s.items():
        data[si][k] = data[si][k].sort(by=["stay_id", "time"])


if __name__ == "__main__":
    setup_logging()
    print(os.listdir(data_dir))
    print(data)
