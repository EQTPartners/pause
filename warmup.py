"""
Copyright (C) eqtgroup.com Ltd 2021
https://github.com/EQTPartners/pause
License: MIT, https://github.com/EQTPartners/pause/LICENSE.md
"""


import tensorflow as tf


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies a warmup schedule on a given learning rate decay schedule."""

    def __init__(
        self,
        initial_learning_rate: float,
        decay_schedule_fn: tf.keras.optimizers.schedules.LearningRateSchedule,
        warmup_steps: int,
        power: float = 1.0,
        name: str = None,
    ) -> None:
        """Initialize the WarmUp Class.

        Args:
            initial_learning_rate (float): initial learning rate.
            decay_schedule_fn (tf.keras.optimizers.schedules.LearningRateSchedule): A learning rate schedule function.
            warmup_steps (int): The number of warm up steps.
            power (float, optional): The power parameter. Defaults to 1.0.
            name (str, optional): The name of the op. Defaults to None.
        """
        super(WarmUp, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step: int) -> tf.Tensor:
        """Obtain the warm-up learning rate.

        Args:
            step (int): The current training step.

        Returns:
            tf.Tensor: The learning rate.
        """
        with tf.name_scope(self.name or "WarmUp") as name:
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(
                warmup_percent_done, self.power
            )
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step),
                name=name,
            )

    def get_config(self) -> dict:
        """Obtain the config of this warm-up object.

        Returns:
            dict: The values of configurations.
        """
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }
