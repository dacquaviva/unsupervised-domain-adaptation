from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
import logging
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import List
from pytorch_lightning.callbacks import LearningRateMonitor
log = logging.getLogger(__name__)


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None: 
    log.info("\n" + OmegaConf.to_yaml(cfg))
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    data_module: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, n_class=cfg.datamodule.n_class, optim = cfg.optim, augmentation=cfg.datamodule.data_augmentation, num_sample_test_embeddings_projector=cfg.datamodule.num_sample_test_embeddings_projector)
    log.info(f"Instantiating logger <{cfg.logger._target_}>")
    logger: pl_loggers.TensorBoardLogger = hydra.utils.instantiate(cfg.logger)
    callbacks: List[Callback] = []
    for _, cb_conf in cfg.callbacks.items():
        log.info(f"Instantiating callback <{cb_conf._target_}>")
        callbacks.append(hydra.utils.instantiate(cb_conf))
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)
    log.info("Starting training!")
    trainer.fit(model, data_module)
    # Print path to best checkpoint
    log.info(f"Last checkpoint path:{trainer.checkpoint_callback.best_model_path}")
    log.info("Starting testing!")
    trainer.test(ckpt_path=None)


if __name__ == '__main__':
    main() 
    