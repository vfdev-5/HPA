import os
import sys
from pathlib import Path
import argparse
from importlib import util
from collections import defaultdict
from functools import partial

import pandas as pd

import tempfile
import random

import logging
import shutil

import torch
from torch.utils.data import DataLoader

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

    logger = logging.getLogger("Validation First Level")
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
    mlflow.set_experiment("Validation First Level" if not config['debug'] else "Debug")
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
            validate(config, logger)
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
from custom_ignite.metrics.accuracy import Accuracy
from custom_ignite.metrics.precision import Precision
from custom_ignite.metrics.recall import Recall
from ignite.metrics import MetricsLambda


def get_object_name(obj):
    return obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__


def prepare_batch(batch, device=None, non_blocking=False):
    x, y = batch['image'], batch['tags']
    return (convert_tensor(x, device=device, non_blocking=non_blocking), y)


def weights_path(client, run_uuid, weights_filename):
    path = Path(client.tracking_uri)
    run_info = client.get_run(run_id=run_uuid)
    artifact_uri = run_info.info.artifact_uri
    artifact_uri = artifact_uri[artifact_uri.find("/") + 1:]
    path /= Path(artifact_uri) / weights_filename
    assert path.exists(), "File is not found at {}".format(path.as_posix())
    return path.as_posix()


def setup_metrics():
    accuracy_metric = Accuracy(is_multilabel=True)
    precision_metric = Precision(average=False, is_multilabel=True)
    recall_metric = Recall(average=False, is_multilabel=True)

    f1_metric = precision_metric * recall_metric * 2 / (recall_metric + precision_metric + 1e-20)
    f1_metric = MetricsLambda(lambda t: torch.mean(t).item(), f1_metric)

    metrics = {
        "accuracy": accuracy_metric,
        "precision": precision_metric,
        "recall": recall_metric,
        "f1": f1_metric
    }

    def thresholded_output_transform_per_tag(output, k):
        y_pred, y = output
        y_pred, y = y_pred[:, k], y[:, k]
        y_pred = torch.round(torch.sigmoid(y_pred))
        return y_pred, y

    # Add Precision/Recall per tag
    for i, t in enumerate(HPADataset.tags):
        metrics['pr_{}'.format(t)] = Precision(
            output_transform=partial(thresholded_output_transform_per_tag, k=i)
        )
        metrics['re_{}'.format(t)] = Recall(
            output_transform=partial(thresholded_output_transform_per_tag, k=i)
        )
    return metrics


def validate(config, logger):

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

    val_loader = config['val_loader']

    y_probas_mean_tta = [0 for _ in range(len(val_loader))]
    y_true = []

    n_tta = config['n_tta']

    @predictor_tta.on(Events.ITERATION_COMPLETED)
    def save_tta_predictions(engine):
        output = engine.state.output
        iteration = (engine.state.iteration - 1) % len(val_loader)
        y_probas = torch.sigmoid(output[0].detach())

        y_probas_mean_tta[iteration] += y_probas * 1.0 / n_tta

        tta_index = engine.state.epoch - 1
        if tta_index == 0:
            y_true.append(output[1])

    logger.info("Start predictions with {} TTA".format(n_tta))
    mlflow.log_param("num_tta", n_tta)
    predictor_tta.run(val_loader, max_epochs=n_tta)
    logger.debug("Ended predictions")

    y_preds_mean_tta = [torch.round(y_probas).cpu() for y_probas in y_probas_mean_tta]

    validator = Engine(lambda engine, batch: batch)
    ProgressBar(desc='Validation').attach(validator)

    metrics = setup_metrics()

    for name, metric in metrics.items():
        metric.attach(validator, name)

    tags_counter = defaultdict(int)

    @validator.on(Events.ITERATION_COMPLETED)
    def count_tags(engine):
        _, y = engine.state.output
        for i, t in enumerate(HPADataset.tags):
            tags_counter[t] += torch.sum(y[:, i]).item()

    logger.info("Compute metrics")
    data = [(y_pred, y) for y_pred, y in zip(y_preds_mean_tta, y_true)]
    validator.run(data, max_epochs=1)

    acc_score = validator.state.metrics['accuracy']
    pr_score = torch.mean(validator.state.metrics['precision']).item()
    re_score = torch.mean(validator.state.metrics['recall']).item()
    f1_score = validator.state.metrics['f1']

    mlflow.log_metric("accuracy", acc_score)
    mlflow.log_metric("precision", pr_score)
    mlflow.log_metric("recall", re_score)
    mlflow.log_metric("f1", f1_score)

    for t in HPADataset.tags:
        mlflow.log_metric("{} Pr".format(t), validator.state.metrics['pr_{}'.format(t)].item())
        mlflow.log_metric("{} Re".format(t), validator.state.metrics['re_{}'.format(t)].item())    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config python file")
    args = parser.parse_args()

    assert "MLFLOW_TRACKING_URI" in os.environ

    config = read_config(args.config)
    config['config_filepath'] = Path(args.config)

    run(config)
