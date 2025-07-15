import numpy as np
from tensorflow.keras import models, layers, optimizers
from PINN_FS import PINNs
from matplotlib import pyplot as plt
from time import time
from train_configs import FS_config
from error import l2norm_err
from loguru import logger

def main() -> None:
    """
    Train and evaluate PINN for the Falkner-Skan boundary layer problem.
    """
    data = np.load('data/Falkner_Skan_Ref_Data.npz')
    velocity_u = data['u'].T
    velocity_v = data['v'].T
    coordinates_x = data['x'].T
    coordinates_y = data['y'].T
    pressure_p = data['p'].T
    coordinates_x -= coordinates_x.min()
    coordinates_y -= coordinates_y.min()
    reference_data = np.stack((velocity_u, velocity_v, pressure_p))
    activation_function = FS_config.act
    neurons_per_layer = FS_config.n_neural
    num_layers = FS_config.n_layer
    adam_steps = FS_config.n_adam
    checkpoint_step = FS_config.cp_step
    boundary_condition_step = FS_config.bc_step
    collection_points = np.concatenate((coordinates_x[:, ::checkpoint_step].reshape((-1, 1)),
                                        coordinates_y[:, ::checkpoint_step].reshape((-1, 1))), axis=1)
    num_collection_points = len(collection_points)
    boundary_condition_flags = np.zeros(coordinates_x.shape, dtype=bool)
    boundary_condition_flags[[0, -1], ::boundary_condition_step] = True
    boundary_condition_flags[:, [0, -1]] = True
    boundary_x = coordinates_x[boundary_condition_flags].flatten()
    boundary_y = coordinates_y[boundary_condition_flags].flatten()
    boundary_u = velocity_u[boundary_condition_flags].flatten()
    boundary_v = velocity_v[boundary_condition_flags].flatten()
    boundary_conditions = np.array([boundary_x, boundary_y, boundary_u, boundary_v]).T
    num_input_vars = 2
    num_output_vars = boundary_conditions.shape[1] - num_input_vars + 1
    pressure_output = 1
    boundary_condition_indices = np.random.choice([False, True], len(boundary_conditions), p=[1 - pressure_output, pressure_output])
    boundary_conditions = boundary_conditions[boundary_condition_indices]
    num_boundary_conditions = len(boundary_conditions)
    test_name = f'_{neurons_per_layer}_{num_layers}_{activation_function}_{adam_steps}_{num_collection_points}_{num_boundary_conditions}'
    input_layer = layers.Input(shape=(num_input_vars,))
    hidden_layer = input_layer
    for _ in range(num_layers):
        hidden_layer = layers.Dense(neurons_per_layer, activation=activation_function)(hidden_layer)
    output_layer = layers.Dense(num_output_vars)(hidden_layer)
    model = models.Model(input_layer, output_layer)
    logger.info(model.summary())
    learning_rate = 1e-3
    optimizer = optimizers.Adam(learning_rate)
    pinn_model = PINNs(model, optimizer, adam_steps)
    logger.info(f"INFO: Start training case : {test_name}")
    start_time = time()
    history = pinn_model.fit(boundary_conditions, collection_points)
    end_time = time()
    training_time = end_time - start_time
    prediction_points = np.array([coordinates_x.flatten(), coordinates_y.flatten()]).T
    predicted_data = pinn_model.predict(prediction_points)
    predicted_u = predicted_data[:, 0].reshape(velocity_u.shape)
    predicted_v = predicted_data[:, 1].reshape(velocity_u.shape)
    predicted_p = predicted_data[:, 2].reshape(velocity_u.shape)
    pressure_shift = pressure_p[0, 0] - predicted_p[0, 0]
    predicted_p += pressure_shift
    predicted_data = np.stack((predicted_u, predicted_v, predicted_p))
    np.savez_compressed('pred/res_FS' + test_name, pred=predicted_data, ref=reference_data,
                        x=coordinates_x, y=coordinates_y, hist=history, err=l2norm_err, ct=training_time)
    model.save('models/model_FS' + test_name + '.h5')
    logger.info("INFO: Prediction and model have been saved!")

if __name__ == "__main__":
    main()
