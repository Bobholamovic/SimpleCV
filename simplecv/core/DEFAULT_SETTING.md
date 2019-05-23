## default settings in simplecv config
### default train config

```python
train=dict(
        forward_times=1,
        # `num_epochs` is mutually exclusive `num_iters`. Please only use one of them
        # Recommend `num_iters` because it is more stable now.
        num_iters=-1,
        num_epochs=-1,
        eval_per_epoch=False,
        tensorboard_interval_step=100,
        log_interval_step=1,
        summary_grads=False,
        summary_weights=False,
        distributed=False,
        # when use apex_ddp_train.py, whether to use apex sync bn.
        apex_sync_bn=False,
        eval_after_train=True,
        resume_from_last=True,
        # when use ddp_train.py, whether to use official nn.SyncBatchNorm.
        sync_bn=False,
        # `normal` or `prefetched`
        iterator_type='normal',
        save_ckpt_interval_epoch=1,
    )
```