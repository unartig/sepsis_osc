import logging
from pathlib import Path

from icu_benchmarks.data.split_process_data import preprocess_data
from icu_benchmarks.constants import RunMode

from sepsis_osc.utils.logger import setup_logging
from sepsis_osc.utils.config import yaib_data_dir
from sepsis_osc.model.gin_configs import file_names, vars, modality_mapping
import os

setup_logging()
logger = logging.getLogger(__name__)

data_dir = Path(yaib_data_dir)

print(os.listdir(data_dir))


data = preprocess_data(
    data_dir=data_dir,
    file_names=file_names,
    seed=666,
    debug=False,
    load_cache=True,
    generate_cache=True,
    cv_repetitions=5,
    repetition_index=0,
    train_size=None,
    cv_folds=5,
    fold_index=0,
    pretrained_imputation_model=None,
    runmode=RunMode.classification,
    vars=vars,
    modality_mapping=modality_mapping,
)


if __name__ == "__main__":
    print(data)
