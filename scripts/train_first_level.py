import os
import sys
from pathlib import Path
import argparse
from importlib import util

import tempfile
import random

import logging
import shutil

import torch
from torch.utils.data import DataLoader

import mlflow

sys.path.insert(0, (Path(__file__).parent.parent / "common").as_posix())
sys.path.insert(0, (Path(__file__).parent.parent / "IgniteConfRunner").as_posix())


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

    logger = logging.getLogger("Training First Level")
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
    mlflow.set_experiment("Training First Level" if not config['debug'] else "Debug")
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
            train(config, logger)
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
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, convert_tensor
from ignite.handlers import ModelCheckpoint, Timer, TerminateOnNan, EarlyStopping
from ignite.metrics import RunningAverage, Loss
from ignite.contrib.handlers import ProgressBar

from custom_ignite.engines.fp16 import create_supervised_fp16_trainer, create_supervised_fp16_evaluator


def get_object_name(obj):
    return obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__


def setup_timer(engine):
    timer = Timer(average=True)
    timer.attach(engine,
                 start=Events.EPOCH_STARTED,
                 resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED)
    return timer


def setup_log_training_loss(trainer, logger, config):

    avg_output = RunningAverage(output_transform=lambda out: out)
    avg_output.attach(trainer, 'running_avg_loss')

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        mlflow.log_metric("training_loss_vs_iterations", engine.state.metrics['running_avg_loss'])
        len_train_dataloader = len(config['train_loader'])
        iteration = (engine.state.iteration - 1) % len_train_dataloader + 1
        if iteration % config['log_interval'] == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.4f}".format(engine.state.epoch, iteration,
                                                                         len_train_dataloader,
                                                                         engine.state.metrics['running_avg_loss']))


def setup_trainer_handlers(trainer, logger, config):
    # Setup timer to measure training time
    timer = setup_timer(trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_time(engine):
        logger.info("One epoch training time (seconds): {}".format(timer.value()))

    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    setup_log_training_loss(trainer, logger, config)

    last_model_saver = ModelCheckpoint(config['log_dir'].as_posix(),
                                       filename_prefix="checkpoint",
                                       save_interval=config['trainer_checkpoint_interval'],
                                       n_saved=1,
                                       atomic=True,
                                       create_dir=True,
                                       save_as_state_dict=True)

    model_name = get_object_name(config['model'])

    to_save = {
        model_name: config['model'],
        "optimizer": config['optimizer'],
    }

    if 'lr_scheduler' in config:
        to_save['lr_scheduler'] = config['lr_scheduler']
    trainer.add_event_handler(Events.ITERATION_COMPLETED, last_model_saver, to_save)

    if 'lr_scheduler' in config:
        @trainer.on(Events.EPOCH_STARTED)
        def update_lr_scheduler(engine):
            config['lr_scheduler'].step()


def setup_log_learning_rate(trainer, logger, config):
    @trainer.on(Events.EPOCH_STARTED)
    def log_lrs(engine):
        if len(config['optimizer'].param_groups) == 1:
            lr = float(config['optimizer'].param_groups[0]['lr'])
            logger.debug("Learning rate: {}".format(lr))
            mlflow.log_metric("learning_rate", lr)
        else:
            for i, param_group in enumerate(config['optimizer'].param_groups):
                lr = float(param_group['lr'])
                logger.debug("Learning rate (group {}): {}".format(i, lr))
                mlflow.log_metric("learning_rate_group_{}".format(i), lr)


def prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options

    """
    x, y = batch['image'], batch['target'].type(torch.float)
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))


def setup_offline_train_metrics_computation(trainer, logger, config):

    train_eval_loader = config['train_eval_loader']
    msg = "- train evaluation data loader: {} number of batches".format(len(train_eval_loader))
    if isinstance(train_eval_loader, DataLoader):
        msg += " | {} number of samples".format(len(train_eval_loader.sampler))
    logger.debug(msg)

    use_fp16 = config['use_fp16'] if 'use_fp16' in config else False
    if not use_fp16:
        train_evaluator = create_supervised_evaluator(config['model'], metrics=config['metrics'],
                                                      prepare_batch=prepare_batch,
                                                      device=config['device'],
                                                      non_blocking="cuda" in config['device'])
    else:
        train_evaluator = create_supervised_fp16_evaluator(config['model'], metrics=config['metrics'],
                                                           prepare_batch=prepare_batch,
                                                           device=config['device'],
                                                           non_blocking="cuda" in config['device'])

    pbar = ProgressBar(desc='Train evaluation')
    pbar.attach(train_evaluator)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_metrics(engine):
        epoch = engine.state.epoch
        if epoch % config['val_interval_epochs'] == 0:
            logger.debug("Compute training metrics")
            metrics_results = train_evaluator.run(train_eval_loader).metrics
            logger.info("Training Results - Epoch: {}".format(epoch))
            for name in config['metrics']:
                logger.info("\tAverage {}: {:.5f}".format(name, metrics_results[name]))
                mlflow.log_metric("training_avg_{}".format(name), metrics_results[name])

    return train_evaluator


def setup_val_metrics_computation(trainer, logger, config):

    use_fp16 = config['use_fp16'] if 'use_fp16' in config else False
    if not use_fp16:
        val_evaluator = create_supervised_evaluator(config['model'], metrics=config['val_metrics'],
                                                    prepare_batch=prepare_batch,
                                                    device=config['device'],
                                                    non_blocking="cuda" in config['device'])
    else:
        val_evaluator = create_supervised_fp16_evaluator(config['model'], metrics=config['val_metrics'],
                                                         prepare_batch=prepare_batch,
                                                         device=config['device'],
                                                         non_blocking="cuda" in config['device'])
    pbar = ProgressBar(desc='Validation')
    pbar.attach(val_evaluator)

    val_dataloader = config['val_loader']

    msg = "- validation data loader: {} number of batches".format(len(val_dataloader))
    if isinstance(val_dataloader, DataLoader):
        msg += " | {} number of samples".format(len(val_dataloader.sampler))
    logger.debug(msg)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        epoch = engine.state.epoch
        if epoch % config['val_interval_epochs'] == 0:
            logger.debug("Compute validation metrics")
            metrics_results = val_evaluator.run(val_dataloader).metrics
            logger.info("Validation Results - Epoch: {}".format(epoch))
            for name in config['val_metrics']:
                logger.info("\tAverage {}: {:.5f}".format(name, metrics_results[name]))
                mlflow.log_metric("validation_avg_{}".format(name), metrics_results[name])

    return val_evaluator


def setup_early_stopping(trainer, config, val_evaluator, score_function):

    if 'early_stopping_kwargs' not in config:
        return

    kwargs = dict(config['early_stopping_kwargs'])
    if 'score_function' not in kwargs:
        kwargs['score_function'] = score_function
    handler = EarlyStopping(trainer=trainer, **kwargs)
    setup_logger(handler._logger, config['log_filepath'], config['log_level'])
    val_evaluator.add_event_handler(Events.COMPLETED, handler)


def setup_best_model_checkpointing(config, val_evaluator, score_function):

    # Setup model checkpoint:
    if 'model_checkpoint_kwargs' not in config:
        config['model_checkpoint_kwargs'] = {
            "filename_prefix": "model",
            "score_name": "val_loss",
            "score_function": score_function,
            "n_saved": 3,
            "atomic": True,
            "create_dir": True,
            "save_as_state_dict": True
        }
    model_name = get_object_name(config['model'])
    best_model_saver = ModelCheckpoint(config['log_dir'].as_posix(),
                                       **config['model_checkpoint_kwargs'])
    val_evaluator.add_event_handler(Events.COMPLETED, best_model_saver, {model_name: config['model']})


def weights_path(client, run_uuid, weights_filename):
    path = Path(client.tracking_uri)
    run_info = client.get_run(run_id=run_uuid)
    artifact_uri = run_info.info.artifact_uri
    artifact_uri = artifact_uri[artifact_uri.find("/") + 1:]
    path /= Path(artifact_uri) / weights_filename
    assert path.exists(), "File is not found at {}".format(path.as_posix())
    return path.as_posix()


def train(config, logger):

    model = config['model']
    criterion = config['criterion']
    optimizer = config['optimizer']

    if 'run_uuid' in config and 'weights_filename' in config:
        run_uuid = config['run_uuid']
        weights_filename = config['weights_filename']
        logger.info("Load weights from {}/{}".format(run_uuid, weights_filename))
        client = mlflow.tracking.MlflowClient(tracking_uri=os.environ['MLFLOW_TRACKING_URI'])
        model.load_state_dict(torch.load(weights_path(client, run_uuid, weights_filename)))

    config['metrics']['loss'] = Loss(criterion)

    mlflow.log_param("model", get_object_name(model))
    mlflow.log_param("criterion", get_object_name(criterion))
    mlflow.log_param("optimizer", get_object_name(optimizer))

    use_fp16 = config['use_fp16'] if 'use_fp16' in config else False

    if not use_fp16:
        trainer = create_supervised_trainer(model, optimizer, criterion,
                                            prepare_batch=prepare_batch,
                                            device=config['device'],
                                            non_blocking="cuda" in config['device'])
    else:
        trainer = create_supervised_fp16_trainer(model, optimizer, criterion,
                                                 prepare_batch=prepare_batch)

    # add typical handlers
    setup_trainer_handlers(trainer, logger, config)
    setup_log_learning_rate(trainer, logger, config)

    num_epochs = config['num_epochs']

    def default_score_function(engine):
        val_loss = engine.state.metrics['loss']
        # Objects with highest scores will be retained.
        return -val_loss

    setup_offline_train_metrics_computation(trainer, logger, config)
    val_evaluator = setup_val_metrics_computation(trainer, logger, config)
    setup_early_stopping(trainer, config, val_evaluator, default_score_function)
    setup_best_model_checkpointing(config, val_evaluator, default_score_function)

    train_dataloader = config['train_loader']

    logger.info("Start training: {} epochs".format(num_epochs))
    mlflow.log_param("num_epochs", num_epochs)
    trainer.run(train_dataloader, max_epochs=num_epochs)
    logger.debug("Training is ended")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config python file")
    args = parser.parse_args()

    assert "MLFLOW_TRACKING_URI" in os.environ

    config = read_config(args.config)
    config['config_filepath'] = Path(args.config)

    run(config)
