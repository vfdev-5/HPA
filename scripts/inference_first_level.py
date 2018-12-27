import os
import sys
from pathlib import Path
import argparse
from importlib import util

import tempfile
import random

import logging
import shutil

import numpy as np
import pandas as pd

import torch

import mlflow

sys.path.insert(0, (Path(__file__).parent.parent / "common").as_posix())
sys.path.insert(0, (Path(__file__).parent.parent / "IgniteConfRunner").as_posix())

from dataflow.datasets import HPADataset


def read_config(filepath):
    """Method to load python configuration file

    Args:
      filepath (str): path to python configuration file

    Returns:
      dictionary

    """
    filepath = Path(filepath)
    assert filepath.exists(), "Configuration file is not found at {}".format(filepath)

    # Load custom module
    spec = util.spec_from_file_location("config", filepath.as_posix())
    custom_module = util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)
    config = custom_module.__dict__
    return config


def clean_config(config, schema_keys):
    """Return a clean module dictionary"""
    new_config = {}
    keys = list(config.keys())
    for k in keys:
        if k in schema_keys:
            new_config[k] = config[k]
    return new_config


def setup_logger(logger, log_filepath=None, level=logging.INFO):

    if logger.hasHandlers():
        for h in list(logger.handlers):
            logger.removeHandler(h)

    logger.setLevel(level)

    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s|%(name)s|%(levelname)s| %(message)s")

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_filepath is not None:
        # create file handler which logs even debug messages
        fh = logging.FileHandler(log_filepath)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def set_seed(seed):
    if seed is None:
        seed = random.randint(0, 10000)

    random.seed(seed)
    torch.manual_seed(seed)


def run(config):

    logger = logging.getLogger("Inference First Level")
    log_level = logging.INFO

    if config['debug']:
        log_level = logging.DEBUG
        print("Activated debug mode")

    log_dir = Path(tempfile.mkdtemp())
    print("Log dir : {}".format(log_dir))
    log_filepath = (log_dir / "run.log").as_posix()
    setup_logger(logger, log_filepath, log_level)
    config['log_dir'] = log_dir
    config['log_filepath'] = log_filepath
    config['log_level'] = log_level

    logger.info("PyTorch version: {}".format(torch.__version__))
    logger.info("Ignite version: {}".format(ignite.__version__))
    logger.info("MLFlow version: {}".format(mlflow.__version__))

    # This sets also experiment id as stated by `mlflow.start_run`
    mlflow.set_experiment("Inference First Level" if not config['debug'] else "Debug")
    source_name = config['config_filepath'].stem
    with mlflow.start_run(source_name=source_name):
        set_seed(config['seed'])
        mlflow.log_param("seed", config['seed'])
        mlflow.log_artifact(config['config_filepath'].as_posix())

        if 'cuda' in config['device']:
            assert torch.cuda.is_available(), \
                "Device {} is not compatible with torch.cuda.is_available()".format(config['device'])
            from torch.backends import cudnn
            cudnn.benchmark = True
            logger.info("CUDA version: {}".format(torch.version.cuda))

        try:
            inference(config, logger)
        except KeyboardInterrupt:
            logger.info("Catched KeyboardInterrupt -> exit")
        except Exception as e:  # noqa
            logger.exception("")
            if config['debug']:
                try:
                    # open an ipython shell if possible
                    import IPython
                    IPython.embed()  # noqa
                except ImportError:
                    print("Failed to start IPython console to debug")

        # Transfer log dir to mlflow
        mlflow.log_artifacts(log_dir.as_posix())

        # Remove temp folder:
        shutil.rmtree(log_dir.as_posix())


import ignite
from ignite.engine import create_supervised_evaluator, convert_tensor, Events, Engine
from ignite.contrib.handlers import ProgressBar


def get_object_name(obj):
    return obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__


def prepare_batch(batch, device=None, non_blocking=False):
    x, x_id = batch['image'], batch['id']
    return convert_tensor(x, device=device, non_blocking=non_blocking), x_id


def weights_path(client, run_uuid, weights_filename):
    path = Path(client.tracking_uri)
    run_info = client.get_run(run_id=run_uuid)
    artifact_uri = run_info.info.artifact_uri
    artifact_uri = artifact_uri[artifact_uri.find("/") + 1:]
    path /= Path(artifact_uri) / weights_filename
    assert path.exists(), "File is not found at {}".format(path.as_posix())
    return path.as_posix()


def inference(config, logger):

    model = config['model']
    run_uuid = config['run_uuid']
    weights_filename = config['weights_filename']
    device = config['device']

    client = mlflow.tracking.MlflowClient(tracking_uri=os.environ['MLFLOW_TRACKING_URI'])
    model.load_state_dict(torch.load(weights_path(client, run_uuid, weights_filename)))

    mlflow.log_param("model", get_object_name(model))
    mlflow.log_param("training run id", run_uuid)
    mlflow.log_param("weights_filename", weights_filename)

    model.to(device)
    _ = model.eval()

    predictor_tta = create_supervised_evaluator(model, device=device, non_blocking="cuda" in device, prepare_batch=prepare_batch)
    ProgressBar(desc='Predict TTA ', persist=True).attach(predictor_tta)

    test_loader = config['test_loader']

    y_probas_mean_tta = np.zeros((len(test_loader.sampler), len(HPADataset.tags)), dtype=np.float32)
    x_ids = []

    n_tta = config['n_tta']
    batch_size = test_loader.batch_size

    @predictor_tta.on(Events.ITERATION_COMPLETED)
    def save_tta_predictions(engine):
        output = engine.state.output
        i = (engine.state.iteration - 1) % len(test_loader)
        y_probas = torch.sigmoid(output[0].detach()).cpu().numpy()
        y_probas_mean_tta[i * batch_size:(i + 1) * batch_size, :] += y_probas * 1.0 / n_tta

        tta_index = engine.state.epoch - 1
        if tta_index == 0:
            x_ids.extend(output[1])

    logger.info("Start predictions with {} TTA".format(n_tta))
    mlflow.log_param("num_tta", n_tta)
    predictor_tta.run(test_loader, max_epochs=n_tta)
    logger.debug("Ended predictions")

    output = pd.DataFrame({"Id": x_ids})
    for i, tag in enumerate(HPADataset.tags):
        output.loc[:, tag] = y_probas_mean_tta[:, i]

    output_filepath = Path(config['log_dir']) / "preds_{}.csv".format(get_object_name(model))
    output.to_csv(output_filepath, index=None)

    if "write_submission" in config and config['write_submission']:
        y_preds_mean_tta = np.round(y_probas_mean_tta)
        output = pd.DataFrame({"Id": x_ids})
        output.loc[:, "Predicted"] = ""

        data = []
        for y in y_preds_mean_tta:
            res = " ".join([str(v) for v in np.where(y > 0)[0]])
            if len(res) < 1:
                res = "0"
            data.append(res)

        output.loc[:, "Predicted"] = data
        output_filepath = Path(config['log_dir']) / "submission_{}.csv".format(get_object_name(model))
        output.to_csv(output_filepath, index=None)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config python file")
    args = parser.parse_args()

    assert "MLFLOW_TRACKING_URI" in os.environ

    config = read_config(args.config)
    config['config_filepath'] = Path(args.config)

    run(config)
