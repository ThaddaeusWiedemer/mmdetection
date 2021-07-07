# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings

import torch

import mmcv
from mmcv.runner.base_runner import BaseRunner
from mmcv.runner.builder import RUNNERS
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.utils import get_host_info


@RUNNERS.register_module()
class EpochBasedRunnerAdaptive(BaseRunner):
    """Epoch-based Runner for domain adaptation.

    This runner train models epoch by epoch with a source and target dataset.
    """

    def run_iter(self, data_batch_src, data_batch_tgt, train_mode, **kwargs):
        if self.batch_processor is not None:
            # based on my current understanding, this should not be reached in adaptive training,
            # as model.train_step is available
            raise NotImplementedError('`EpochBasedRunnerAdaptive.run_iter` called `batch_processor`, \
                which cannot handle source/target domains yet')
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            # concat src and tgt batches to guarantee correct subsequent scattering to GPUs
            # TODO implementation here really depends on what type the data batches have
            # try simple tuple and see what happends
            print('data batches in `Runner.run_iter()` have type ', type(data_batch_src))
            data_batch = (data_batch_src, data_batch_tgt)
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            # only use target domain data
            outputs = self.model.val_step(data_batch_tgt, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader_src, data_loader_tgt, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader_src = data_loader_src
        self.data_loader_tgt = data_loader_tgt
        # TODO how to handle this if dataloaders are not of equal size?
        self._max_iters = self._max_epochs * len(self.data_loader_tgt)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        # TODO can be implemented better with zip() if unequal dataloader lengths are handled
        for i, data_batch_tgt in enumerate(self.data_loader_tgt):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(self.data_loader_src[i], data_batch_tgt, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader_src, data_loader_tgt, **kwargs):
        # only use target domain for validation
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader_tgt
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders_src, data_loaders_tgt, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders_src, list)
        assert isinstance(data_loaders_tgt, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders_src) == len(workflow)
        assert len(data_loaders_tgt) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                # TODO how to handle this if data loaders are not equal size? can we reuse source samples?
                self._max_iters = self._max_epochs * len(data_loaders_tgt[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders_src[i], data_loaders_tgt[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)

                