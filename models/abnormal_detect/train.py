"""Anomalib Traning Script.

This script reads the name of the model or config file from command
line, train/test the anomaly model to get quantitative and qualitative
results.
"""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import logging
import warnings
from argparse import ArgumentParser, Namespace
import json
import os
import numpy as np

from pytorch_lightning import Trainer, seed_everything
from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from anomalib.utils.loggers import configure_logger, get_experiment_logger
# from anomalib.deploy.optimize import get_model_metadata

logger = logging.getLogger("anomalib")

def get_args() -> Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, help="Name of the algorithm to train/test",
                        default="patchcore"
                        # default='fastflow'
                        )
    parser.add_argument("--config", type=str, help="Path to a model config file",
                        default='/home/quan/Desktop/company/Reconstruct3D_Pipeline/models/abnormal_detect/cfg/patchcore_2.yaml'
                        # default='/home/quan/Desktop/company/Reconstruct3D_Pipeline/models/abnormal_detect/cfg/fastflow.yaml'
                        )
    parser.add_argument("--log-level", type=str, default="INFO", help="<DEBUG, INFO, WARNING, ERROR>")

    args = parser.parse_args()
    return args

def get_model_metadata(model):
    meta_data = {}
    cached_meta_data = {
        "image_threshold": model.image_threshold.cpu().value.numpy(),
        "pixel_threshold": model.pixel_threshold.cpu().value.numpy(),
        "pixel_mean": model.training_distribution.pixel_mean.cpu().numpy(),
        "image_mean": model.training_distribution.image_mean.cpu().numpy(),
        "pixel_std": model.training_distribution.pixel_std.cpu().numpy(),
        "image_std": model.training_distribution.image_std.cpu().numpy(),
        "min": model.min_max.min.cpu().numpy(),
        "max": model.min_max.max.cpu().numpy(),
    }
    # Remove undefined values by copying in a new dict
    for key, val in cached_meta_data.items():
        if not np.isinf(val).all():
            meta_data[key] = float(val)
    del cached_meta_data
    return meta_data

def train():
    """Train an anomaly classification or segmentation model based on a provided configuration file."""
    args = get_args()
    configure_logger(level=args.log_level)

    if args.log_level == "ERROR":
        warnings.filterwarnings("ignore")

    config = get_configurable_parameters(model_name=args.model, config_path=args.config)
    if config.project.seed != 0:
        seed_everything(config.project.seed)

    datamodule = get_datamodule(config)
    model = get_model(config)
    experiment_logger = get_experiment_logger(config)
    callbacks = get_callbacks(config)

    trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=callbacks)
    logger.info("Training the model.")
    trainer.fit(model=model, datamodule=datamodule)

    logger.info("Loading the best model weights.")
    load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)
    trainer.callbacks.insert(0, load_model_callback)

    logger.info("Testing the model.")
    trainer.test(model=model, datamodule=datamodule)

    meta_data = get_model_metadata(model)
    print(meta_data)
    meta_data_path = os.path.join(config.project.path, 'meta_data.json')
    with open(meta_data_path, 'w') as f:
        json.dump(meta_data, f)


if __name__ == "__main__":
    train()
