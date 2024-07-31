import tensorflow as tf
import os
import numpy as np

class Linear_QNet(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()
        self.model = tf.keras.Sequential([
            # tf.keras.layers.Flatten(input_shape=(input_size,)),
            tf.keras.layers.Dense(units= input_size, activation = "relu"),
            tf.keras.layers.Dense(units= hidden_size, activation="relu"),
            tf.keras.layers.Dense(units= output_size)
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')

    def call(self, x):
        
        x=np.reshape(x, (-1,11))
        return self.model(x)
    
    def save(self, file_name='model.weights.h5'):
        model_folder_path = '\\Users\\daari\\Documents\\Daarius Ryan\\Python\\Snake'
        file_path = os.path.join(model_folder_path, file_name)
        self.save_weights(file_path)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def train_step(self, state, action, reward, next_state, done):
        state = tf.convert_to_tensor(state, dtype=tf.float32)  # Convert state to float32
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.int32)  # Convert action to int32 for one-hot encoding
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        
        if len(state.shape) == 1:
            state = tf.expand_dims(state, 0)
            next_state = tf.expand_dims(next_state, 0)
            action = tf.expand_dims(action, 0)
            reward = tf.expand_dims(reward, 0)
            done = (done, )

        with tf.GradientTape() as tape:
            # 1: predicted Q values with current state
            pred = self.model(state, training=True)
            
            # Create a target tensor with the same shape as pred
            target = tf.identity(pred)

            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    Q_new = reward[idx] + self.gamma * tf.reduce_max(self.model(next_state[idx], training=False))

                # Create a mask to update the target tensor
                action_idx = tf.argmax(action[idx])
                target = tf.tensor_scatter_nd_update(target, [[idx, action_idx]], [Q_new])

            # Compute loss
            loss = self.loss_fn(target, pred)
        
        # 2: Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))