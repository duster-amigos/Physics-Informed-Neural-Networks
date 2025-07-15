import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from lbfgs import optimizer as lbfgs_op
from loguru import logger
from typing import Any

class PINNs(models.Model):
    """
    Physics-Informed Neural Network (PINN) for the Falkner-Skan boundary layer problem.
    Approximates the solution to the Navier-Stokes equations:
        \[
        f_1 = U U_x + V U_y + P_x - \nu (U_{xx} + U_{yy})
        \]
        \[
        f_2 = P_y
        \]
        \[
        f_3 = U_x + V_y
        \]
    Args:
        model: Keras model.
        optimizer: Keras optimizer.
        epochs: Number of training epochs.
    """
    def __init__(self, model: Any, optimizer: Any, epochs: int, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.hist = []
        self.epoch = 0
        self.sopt = lbfgs_op(self.trainable_variables)
        self.nu = 0.01
        logger.info("PINNs model initialized.")

    @tf.function
    def net_f(self, cp: tf.Tensor) -> tf.Tensor:
        """
        Compute the Navier-Stokes residuals for the collocation points.
        Args:
            cp (tf.Tensor): Collocation points.
        Returns:
            tf.Tensor: Residuals.
        """
        cp = self.scalex_r(cp)
        x = cp[:, 0]
        y = cp[:, 1]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            X = tf.stack([x, y], axis=-1)
            X = self.scalex(X)
            pred = self.model(X)
            pred = self.scale_r(pred)
            U = pred[:, 0]
            V = pred[:, 1]
            P = pred[:, 2]
            U_x = tape.gradient(U, x)
            U_y = tape.gradient(U, y)
            V_x = tape.gradient(V, x)
            V_y = tape.gradient(V, y)
        U_xx = tape.gradient(U_x, x)
        U_yy = tape.gradient(U_y, y)
        V_xx = tape.gradient(V_x, x)
        V_yy = tape.gradient(V_y, y)
        P_x = tape.gradient(P, x)
        P_y = tape.gradient(P, y)
        f1 = U * U_x + V * U_y + P_x - self.nu * (U_xx + U_yy)
        f2 = P_y
        f3 = U_x + V_y
        f = tf.stack([f1, f2, f3], axis=-1)
        return f

    @tf.function
    def train_step(self, bc: tf.Tensor, cp: tf.Tensor):
        X = bc[:, :2]
        y = bc[:, 2:]
        with tf.GradientTape() as tape:
            u_p_bc = self.model(X)[:, :-1]
            f = self.net_f(cp)
            loss_bc = tf.reduce_mean(tf.square(y - u_p_bc))
            loss_f = tf.reduce_mean(tf.square(f))
            loss_u = loss_bc
            loss = loss_u + loss_f
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        l1 = tf.reduce_mean(loss)
        l2 = tf.reduce_mean(loss_u)
        l3 = tf.reduce_mean(loss_f)
        logger.info(f"loss: {l1.numpy()}, loss_u: {l2.numpy()}, loss_f: {l3.numpy()}")
        return loss, grads, tf.stack([l1, l2, l3])

    def fit_scale(self, y: tf.Tensor) -> tf.Tensor:
        ymax = tf.reduce_max(tf.abs(y), axis=0)
        self.ymax = tf.concat([ymax, [1.0]], 0)
        return y / ymax

    @tf.function
    def scale(self, y: tf.Tensor) -> tf.Tensor:
        return y / self.ymax

    @tf.function
    def scale_r(self, ys: tf.Tensor) -> tf.Tensor:
        return ys * self.ymax

    def fit_scalex(self, x: tf.Tensor) -> tf.Tensor:
        xmax = tf.reduce_max(tf.abs(x), axis=0)
        xmin = tf.reduce_min(x, axis=0)
        self.xmax = xmax
        self.xmin = xmin
        xs = ((x - xmin) / (xmax - xmin))
        return xs

    @tf.function
    def scalex(self, x: tf.Tensor) -> tf.Tensor:
        xs = ((x - self.xmin) / (self.xmax - self.xmin))
        return xs

    @tf.function
    def scalex_r(self, xs: tf.Tensor) -> tf.Tensor:
        x = (xs) * (self.xmax - self.xmin) + self.xmin
        return x

    def fit(self, bc: np.ndarray, cp: np.ndarray) -> np.ndarray:
        bc = tf.convert_to_tensor(bc, tf.float32)
        cp = tf.convert_to_tensor(cp, tf.float32)
        x_bc = bc[:, :2]
        y_bc = bc[:, 2:]
        y_bc = self.fit_scale(y_bc)
        x_bc = self.fit_scalex(x_bc)
        cp = self.scalex(cp)
        bc = tf.concat([x_bc, y_bc], axis=1)
        def func(params_1d):
            self.sopt.assign_params(params_1d)
            logger.info(f"epoch: {self.epoch}")
            loss, grads, hist = self.train_step(bc, cp)
            grads = tf.dynamic_stitch(self.sopt.idx, grads)
            self.epoch += 1
            self.hist.append(hist.numpy())
            return loss.numpy().astype(np.float64), grads.numpy().astype(np.float64)
        for epoch in range(self.epochs):
            logger.info(f"epoch: {self.epoch}")
            loss, grads, hist = self.train_step(bc, cp)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.epoch += 1
            self.hist.append(hist.numpy())
        self.sopt.minimize(func)
        return np.array(self.hist)

    def predict(self, cp: np.ndarray) -> np.ndarray:
        cp = tf.convert_to_tensor(cp, tf.float32)
        cp = self.scalex(cp)
        u_p = self.model(cp)
        u_p = self.scale_r(u_p)
        return u_p.numpy()
