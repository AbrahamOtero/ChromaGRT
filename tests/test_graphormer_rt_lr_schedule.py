import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAIRSEQ_ROOT = (
    PROJECT_ROOT
    / "src"
    / "benchmarks"
    / "graphormer_rt"
    / "upstream"
    / "fairseq"
)
sys.path.insert(0, str(FAIRSEQ_ROOT))

from fairseq.optim.lr_scheduler.polynomial_decay_schedule import (  # noqa: E402
    PolynomialDecayLRSchedule,
)
from fairseq_cli.train import (  # noqa: E402
    EarlyStoppingState,
    _configure_fold_scaled_schedule,
    _fold_scaled_schedule,
    should_stop_early,
)


class _Optimizer:
    def __init__(self):
        self.lr = None

    def set_lr(self, lr):
        self.lr = lr

    def get_lr(self):
        return self.lr


class _EpochIterator:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


class _Trainer:
    def __init__(self, scheduler, num_updates=0):
        self.lr_scheduler = scheduler
        self.num_updates = num_updates

    def get_num_updates(self):
        return self.num_updates


def _scheduler(cfg=None):
    scheduler = object.__new__(PolynomialDecayLRSchedule)
    scheduler.cfg = cfg or SimpleNamespace(warmup_updates=1, total_num_update=2)
    scheduler.optimizer = _Optimizer()
    scheduler.lr = 1e-4
    scheduler.end_learning_rate = 0.0
    scheduler.total_num_update = 2
    scheduler.power = 1.0
    scheduler.warmup_factor = 1.0
    return scheduler


def _config(*, patience=3, patience_after_warmup=True, warmup_updates=10):
    return SimpleNamespace(
        checkpoint=SimpleNamespace(
            patience=patience,
            patience_after_warmup=patience_after_warmup,
            maximize_best_checkpoint_metric=False,
            reset_optimizer=False,
            reset_meters=False,
        ),
        lr_scheduler=SimpleNamespace(warmup_updates=warmup_updates),
    )


class FoldScaledScheduleTests(unittest.TestCase):
    def test_schedule_uses_all_epochs_and_update_frequencies(self):
        warmup, total = _fold_scaled_schedule(
            num_batches=10,
            update_freq=[1, 2],
            max_epoch=3,
            warmup_ratio=0.15,
        )
        self.assertEqual(total, 20)
        self.assertEqual(warmup, 3)

    def test_schedule_accounts_for_dropped_remainder(self):
        warmup, total = _fold_scaled_schedule(
            num_batches=10,
            update_freq=[4],
            max_epoch=2,
            warmup_ratio=0.25,
            skip_remainder=True,
        )
        self.assertEqual(total, 4)
        self.assertEqual(warmup, 1)

    def test_reconfigured_scheduler_matches_linear_rule(self):
        scheduler = _scheduler()
        scheduler.reconfigure_schedule(15, 100, num_updates=0)
        self.assertEqual(scheduler.cfg.warmup_updates, 15)
        self.assertEqual(scheduler.cfg.total_num_update, 100)
        self.assertEqual(scheduler.optimizer.lr, 0.0)

        scheduler.step_update(15)
        self.assertEqual(scheduler.optimizer.lr, 1e-4)
        scheduler.step_update(50)
        expected = 1e-4 * (1 - (50 - 15) / (100 - 15))
        self.assertTrue(math.isclose(scheduler.optimizer.lr, expected))
        scheduler.step_update(100)
        self.assertEqual(scheduler.optimizer.lr, 0.0)

    def test_invalid_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            _fold_scaled_schedule(10, [1], 250, 1.0)

    def test_runtime_configuration_uses_the_real_iterator_length(self):
        scheduler = _scheduler()
        trainer = _Trainer(scheduler, num_updates=7)
        cfg = SimpleNamespace(
            lr_scheduler=SimpleNamespace(
                _name="polynomial_decay",
                scale_to_max_epoch=True,
                warmup_ratio=0.15,
            ),
            optimization=SimpleNamespace(
                max_epoch=250,
                update_freq=[1],
                skip_remainder_batch=False,
            ),
        )
        warmup, total = _configure_fold_scaled_schedule(
            cfg, trainer, _EpochIterator(490)
        )
        self.assertEqual(total, 122_500)
        self.assertEqual(warmup, 18_375)
        self.assertEqual(cfg.lr_scheduler.total_num_update, total)
        self.assertEqual(cfg.lr_scheduler.warmup_updates, warmup)
        self.assertEqual(scheduler.cfg.total_num_update, total)
        self.assertEqual(scheduler.cfg.warmup_updates, warmup)
        self.assertTrue(
            math.isclose(scheduler.optimizer.lr, 1e-4 * 7 / warmup)
        )

    def test_real_omegaconf_and_scheduler_copy_remain_synchronized(self):
        cfg = OmegaConf.create(
            {
                "lr_scheduler": {
                    "_name": "polynomial_decay",
                    "scale_to_max_epoch": True,
                    "warmup_ratio": 0.15,
                    "warmup_updates": 33_281,
                    "total_num_update": 221_875,
                },
                "optimization": {
                    "max_epoch": 250,
                    "update_freq": [1],
                    "skip_remainder_batch": False,
                },
                "checkpoint": {
                    "patience": 30,
                    "patience_after_warmup": True,
                    "maximize_best_checkpoint_metric": False,
                },
            }
        )
        scheduler_cfg = OmegaConf.create(
            {
                "warmup_updates": 33_281,
                "total_num_update": 221_875,
            }
        )
        scheduler = _scheduler(scheduler_cfg)
        trainer = _Trainer(scheduler)

        warmup, total = _configure_fold_scaled_schedule(
            cfg, trainer, _EpochIterator(470)
        )

        self.assertEqual((warmup, total), (17_625, 117_500))
        self.assertEqual(cfg.lr_scheduler.warmup_updates, warmup)
        self.assertEqual(cfg.lr_scheduler.total_num_update, total)
        self.assertEqual(scheduler.cfg.warmup_updates, warmup)
        self.assertEqual(scheduler.cfg.total_num_update, total)
        serialized = OmegaConf.to_container(cfg, resolve=True)
        self.assertEqual(serialized["lr_scheduler"]["warmup_updates"], warmup)
        self.assertEqual(serialized["lr_scheduler"]["total_num_update"], total)

        state = EarlyStoppingState(best=1.0)
        self.assertFalse(should_stop_early(cfg, 2.0, warmup, state))
        self.assertEqual(state.num_runs, 0)
        self.assertFalse(should_stop_early(cfg, 2.0, warmup + 1, state))
        self.assertEqual(state.num_runs, 1)

    def test_other_schedulers_keep_their_original_behavior(self):
        cfg = SimpleNamespace(lr_scheduler=SimpleNamespace(_name="fixed"))
        self.assertIsNone(
            _configure_fold_scaled_schedule(cfg, object(), _EpochIterator(10))
        )


class WarmupAwareEarlyStoppingTests(unittest.TestCase):
    def test_patience_is_not_counted_during_warmup(self):
        cfg = _config()
        state = EarlyStoppingState()
        self.assertFalse(should_stop_early(cfg, 1.0, 1, state))
        for update in range(2, 11):
            self.assertFalse(should_stop_early(cfg, 2.0, update, state))
        self.assertEqual(state.best, 1.0)
        self.assertEqual(state.num_runs, 0)

        self.assertFalse(should_stop_early(cfg, 2.0, 11, state))
        self.assertFalse(should_stop_early(cfg, 2.0, 12, state))
        self.assertTrue(should_stop_early(cfg, 2.0, 13, state))

    def test_improvement_after_warmup_resets_patience(self):
        cfg = _config()
        state = EarlyStoppingState(best=1.0)
        self.assertFalse(should_stop_early(cfg, 2.0, 11, state))
        self.assertFalse(should_stop_early(cfg, 0.5, 12, state))
        self.assertEqual(state.best, 0.5)
        self.assertEqual(state.num_runs, 0)

    def test_original_counting_is_preserved_when_gate_is_disabled(self):
        cfg = _config(patience=2, patience_after_warmup=False)
        state = EarlyStoppingState(best=1.0)
        self.assertFalse(should_stop_early(cfg, 2.0, 1, state))
        self.assertTrue(should_stop_early(cfg, 2.0, 2, state))

    def test_state_is_restored_from_checkpoint(self):
        cfg = _config()
        restored = EarlyStoppingState.from_checkpoint(
            cfg,
            {"best": 0.9, "early_stopping": {"best": 0.8, "num_runs": 12}},
        )
        self.assertEqual(restored.best, 0.8)
        self.assertEqual(restored.num_runs, 12)


if __name__ == "__main__":
    unittest.main()
