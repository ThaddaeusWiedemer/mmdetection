import torch

from ..builder import DETECTORS
from .base import BaseDetector

from mmcv.runner import auto_fp16

@DETECTORS.register_module()
class BaseDetectorAdaptive(BaseDetector):
    """Base class for detectors trained using domain adaptation, i.e. source and target domain"""

    def __init__(self, init_cfg=None):
        super(BaseDetector, self).__init__(init_cfg)

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        For domain adaptation, the source and target domain data that was bundled
        for scattering across GPUs needs to be unpacked here.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        # TODO whether we can just pass all the data to forward() depends on the exact packing method in
        # EpochBasedRunnerAdapative.run_iter() and whether pre-forward hooks are defined that need to be dealt with

        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, img_tgt=None, img_metas_tgt=None, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        When ``return_loss=True``, img_tgt and img_metas_tgt can be passed
        for domain adapatation. Otherwise (i.e. validation mode), only a
        single input can be passed.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            # we always want to do domain adaptation
            assert img_tgt is not None and img_metas_tgt is not None, \
                'domain adaptation training needs inputs from target domain'
            return self.forward_train(img, img_metas, img_tgt, img_metas_tgt, **kwargs)
        else:
            # validation is always only using data from one domain
            assert img_tgt is None and img_metas_tgt is None, 'validation and testing only use data from one domain'
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(self, imgs, img_metas, imgs_tgt, img_metas_tgt, **kwargs):
        """
        Args:
            img (list[Tensor]): List of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
            imgs_tgt : same as ``imgs`` for target domain
            img_metas_tgt : same as ``img_metas`` for target domain
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        batch_input_shape = tuple(imgs[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        batch_input_shape = tuple(imgs_tgt[0].size()[-2:])
        for img_meta in img_metas_tgt:
            img_meta['batch_input_shape'] = batch_input_shape

