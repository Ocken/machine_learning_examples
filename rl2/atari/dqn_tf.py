# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import copy
import gym
import os
import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from scipy.misc import imresize

<<<<<<< HEAD
if '../cartpole' not in sys.path:
  sys.path.append('../cartpole')
from q_learning_bins import plot_running_avg

# constants
IM_WIDTH = 80
IM_HEIGHT = 80

# globals
global_step = 0
=======




##### testing only
# MAX_EXPERIENCES = 10000
# MIN_EXPERIENCES = 1000


MAX_EXPERIENCES = 500000
MIN_EXPERIENCES = 50000
TARGET_UPDATE_PERIOD = 10000
IM_SIZE = 80
K = 4 #env.action_space.n


>>>>>>> upstream/master


def downsample_image(A):
  B = A[31:195] # select the important parts of the image
  B = B.mean(axis=2) # convert to grayscale
<<<<<<< HEAD
  # B = B / 255.0 # scale to 0..1
  # imresize scales it back to 255
=======
>>>>>>> upstream/master

  # downsample image
  # changing aspect ratio doesn't significantly distort the image
  # nearest neighbor interpolation produces a much sharper image
  # than default bilinear
<<<<<<< HEAD
  B = imresize(B, size=(IM_HEIGHT, IM_WIDTH), interp='nearest')
  return B


class DQN:
  def __init__(self, K, conv_layer_sizes, hidden_layer_sizes, gamma, scope, max_experiences=500000, min_experiences=50000, batch_sz=32):
=======
  B = imresize(B, size=(IM_SIZE, IM_SIZE), interp='nearest')
  return B


def update_state(state, obs):
  obs_small = downsample_image(obs)
  return np.append(state[1:], np.expand_dims(obs_small, 0), axis=0)


class DQN:
  def __init__(self, K, conv_layer_sizes, hidden_layer_sizes, gamma, scope):

>>>>>>> upstream/master
    self.K = K
    self.scope = scope

    with tf.variable_scope(scope):

      # inputs and targets
<<<<<<< HEAD
      self.X = tf.placeholder(tf.float32, shape=(None, 4, IM_HEIGHT, IM_WIDTH), name='X')
=======
      self.X = tf.placeholder(tf.float32, shape=(None, 4, IM_SIZE, IM_SIZE), name='X')

>>>>>>> upstream/master
      # tensorflow convolution needs the order to be:
      # (num_samples, height, width, "color")
      # so we need to tranpose later
      self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
      self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

      # calculate output and cost
      # convolutional layers
      # these built-in layers are faster and don't require us to
      # calculate the size of the output of the final conv layer!
      Z = self.X / 255.0
      Z = tf.transpose(Z, [0, 2, 3, 1])
      for num_output_filters, filtersz, poolsz in conv_layer_sizes:
        Z = tf.contrib.layers.conv2d(
          Z,
          num_output_filters,
          filtersz,
          poolsz,
          activation_fn=tf.nn.relu
        )

      # fully connected layers
      Z = tf.contrib.layers.flatten(Z)
      for M in hidden_layer_sizes:
        Z = tf.contrib.layers.fully_connected(Z, M)

      # final output layer
      self.predict_op = tf.contrib.layers.fully_connected(Z, K)

      selected_action_values = tf.reduce_sum(
        self.predict_op * tf.one_hot(self.actions, K),
        reduction_indices=[1]
      )

      cost = tf.reduce_mean(tf.square(self.G - selected_action_values))
      # self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)
      # self.train_op = tf.train.AdagradOptimizer(1e-2).minimize(cost)
<<<<<<< HEAD
      # self.train_op = tf.train.RMSPropOptimizer(2.5e-4, decay=0.99, epsilon=10e-3).minimize(cost)
=======
      # self.train_op = tf.train.RMSPropOptimizer(2.5e-4, decay=0.99, epsilon=1e-3).minimize(cost)
>>>>>>> upstream/master
      self.train_op = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(cost)
      # self.train_op = tf.train.MomentumOptimizer(1e-3, momentum=0.9).minimize(cost)
      # self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)

<<<<<<< HEAD
    # create replay memory
    self.experience = []
    self.max_experiences = max_experiences
    self.min_experiences = min_experiences
    self.batch_sz = batch_sz
    self.gamma = gamma
=======
      self.cost = cost
>>>>>>> upstream/master

  def copy_from(self, other):
    mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
    mine = sorted(mine, key=lambda v: v.name)
    theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
    theirs = sorted(theirs, key=lambda v: v.name)

    ops = []
    for p, q in zip(mine, theirs):
      actual = self.session.run(q)
      op = p.assign(actual)
      ops.append(op)

    self.session.run(ops)

  def set_session(self, session):
    self.session = session

<<<<<<< HEAD
  def predict(self, X):
    return self.session.run(self.predict_op, feed_dict={self.X: X})

  def is_training(self):
    return len(self.experience) >= self.min_experiences

  def train(self, target_network):
    # sample a random batch from buffer, do an iteration of GD
    if not self.is_training():
      # don't do anything if we don't have enough experience
      return

    # randomly select a batch
    sample = random.sample(self.experience, self.batch_sz)
    states, actions, rewards, next_states, dones = map(np.array, zip(*sample))
    next_Q = np.max(target_network.predict(next_states), axis=1)
    # targets = [r + self.gamma*next_q if done is False else r for r, next_q, done in zip(rewards, next_Q, dones)]
    targets = rewards + np.invert(dones) * self.gamma * next_Q

    # call optimizer
    self.session.run(
      self.train_op,
=======
  def predict(self, states):
    return self.session.run(self.predict_op, feed_dict={self.X: states})

  def update(self, states, actions, targets):
    c, _ = self.session.run(
      [self.cost, self.train_op],
>>>>>>> upstream/master
      feed_dict={
        self.X: states,
        self.G: targets,
        self.actions: actions
      }
    )
<<<<<<< HEAD

  def add_experience(self, s, a, r, s2, done):
    if len(self.experience) >= self.max_experiences:
      self.experience.pop(0)
    if len(s) != 4 or len(s2) != 4:
      print("BAD STATE")

    # make copies
    s = copy.copy(s)
    s2 = copy.copy(s2)

    self.experience.append((s, a, r, s2, done))
=======
    return c
>>>>>>> upstream/master

  def sample_action(self, x, eps):
    if np.random.random() < eps:
      return np.random.choice(self.K)
    else:
      return np.argmax(self.predict([x])[0])


<<<<<<< HEAD
def update_state(state, observation):
  # downsample and grayscale observation
  observation_small = downsample_image(observation)
  
  # list method
  # state.append(observation_small)
  # if len(state) > 4:
  #   state.pop(0)
  # return state

  # numpy method
  # expect state to be of shape (4,80,80)
  # print("state.shape:", state.shape)
  # B = np.expand_dims(observation_small, 0)
  # print("B.shape:", B.shape)
  return np.append(state[1:], np.expand_dims(observation_small, 0), axis=0)


def play_one(env, model, tmodel, eps, eps_step, gamma, copy_period):
  global global_step

  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0
  # state = []
  # prev_state = []
  # update_state(state, observation) # add the first observation
  state = [downsample_image(observation)]*4
  # prev_state = np.copy(state)
  prev_state = None

  total_time_training = 0
  n_training_steps = 0


  while not done and iters < 2000:
    # if we reach 2000, just quit, don't want this going forever
    # the 200 limit seems a bit early

    if len(state) < 4:
      # we can't choose an action based on model
      action = env.action_space.sample()
    else:
      action = model.sample_action(state, eps)

    # copy state to prev state
    # prev_state.append(state[-1])
    # if len(prev_state) > 4:
    #   prev_state.pop(0)
    prev_state = np.copy(state)

    # perform the action
    observation, reward, done, info = env.step(action)

    # add the new frame to the state
    state = update_state(state, observation)

    totalreward += reward

    # update the model
    if len(state) == 4 and len(prev_state) == 4:
      model.add_experience(prev_state, action, reward, state, done)

      t0 = datetime.now()
      model.train(tmodel)

      if model.is_training():
        dt = (datetime.now() - t0).total_seconds()
        total_time_training += dt
        n_training_steps += 1
        eps = max(eps - eps_step, 0.1)

    iters += 1

    if global_step % copy_period == 0:
      tmodel.copy_from(model)
    global_step += 1

  if n_training_steps > 0:
    print("Training time per step:", total_time_training / n_training_steps)

  return totalreward, eps, iters


def main():
  env = gym.make('Breakout-v0')
  gamma = 0.99
  copy_period = 10000

  D = len(env.observation_space.sample())
  K = 4 #env.action_space.n ### NO! returns 6 but only 4 valid actions
  conv_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
  hidden_sizes = [512]
  model = DQN(K, conv_sizes, hidden_sizes, gamma, scope='main')
  tmodel = DQN(K, conv_sizes, hidden_sizes, gamma, scope='target')
  init = tf.global_variables_initializer()
  session = tf.InteractiveSession()
  session.run(init)
  model.set_session(session)
  tmodel.set_session(session)


  if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)


  N = 100000
  totalrewards = np.empty(N)
  costs = np.empty(N)
  n_max = 500000 # last step to decrease epsilon
  eps_step = 0.9 / n_max
  eps = 1.0
  for n in range(N):
    t0 = datetime.now()
    totalreward, eps, num_steps = play_one(env, model, tmodel, eps, eps_step, gamma, copy_period)
    totalrewards[n] = totalreward
    if n % 1 == 0:
      print("episode:", n, "total reward:", totalreward, "eps:", "%.3f" % eps, "num steps:", num_steps, "episode duration:", (datetime.now() - t0), "avg reward (last 100):", "%.3f" % totalrewards[max(0, n-100):(n+1)].mean())

  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  print("total steps:", totalrewards.sum())

  plt.plot(totalrewards)
  plt.title("Rewards")
  plt.show()

  plot_running_avg(totalrewards)


if __name__ == '__main__':
  main()
=======
def learn(model, target_model, experience_replay_buffer, gamma, batch_size):
  # Sample experiences
  samples = random.sample(experience_replay_buffer, batch_size)
  states, actions, rewards, next_states, dones = map(np.array, zip(*samples))

  # Calculate targets
  next_Qs = target_model.predict(next_states)
  next_Q = np.amax(next_Qs, axis=1)
  targets = rewards + np.invert(dones).astype(np.float32) * gamma * next_Q

  # Update model
  loss = model.update(states, actions, targets)
  return loss


def play_one(
  env,
  total_t,
  experience_replay_buffer,
  model,
  target_model,
  gamma,
  batch_size,
  epsilon,
  epsilon_change,
  epsilon_min):

  t0 = datetime.now()

  # Reset the environment
  obs = env.reset()
  obs_small = downsample_image(obs)
  state = np.stack([obs_small] * 4, axis=0)
  assert(state.shape == (4, 80, 80))
  loss = None


  total_time_training = 0
  num_steps_in_episode = 0
  episode_reward = 0

  done = False
  while not done:

    # Update target network
    if total_t % TARGET_UPDATE_PERIOD == 0:
      target_model.copy_from(model)
      print("Copied model parameters to target network. total_t = %s, period = %s" % (total_t, TARGET_UPDATE_PERIOD))


    # Take action
    action = model.sample_action(state, epsilon)
    obs, reward, done, _ = env.step(action)
    obs_small = downsample_image(obs)
    next_state = np.append(state[1:], np.expand_dims(obs_small, 0), axis=0)
    # assert(state.shape == (4, 80, 80))



    episode_reward += reward

    # Remove oldest experience if replay buffer is full
    if len(experience_replay_buffer) == MAX_EXPERIENCES:
      experience_replay_buffer.pop(0)

    # Save the latest experience
    experience_replay_buffer.append((state, action, reward, next_state, done))

    # Train the model, keep track of time
    t0_2 = datetime.now()
    loss = learn(model, target_model, experience_replay_buffer, gamma, batch_size)
    dt = datetime.now() - t0_2

    total_time_training += dt.total_seconds()
    num_steps_in_episode += 1


    state = next_state
    total_t += 1

    epsilon = max(epsilon - epsilon_change, epsilon_min)

  return total_t, episode_reward, (datetime.now() - t0), num_steps_in_episode, total_time_training/num_steps_in_episode, epsilon



if __name__ == '__main__':

  # hyperparams and initialize stuff
  conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
  hidden_layer_sizes = [512]
  gamma = 0.99
  batch_sz = 32
  num_episodes = 10000
  total_t = 0
  experience_replay_buffer = []
  episode_rewards = np.zeros(num_episodes)



  # epsilon
  # decays linearly until 0.1
  epsilon = 1.0
  epsilon_min = 0.1
  epsilon_change = (epsilon - epsilon_min) / 500000



  # Create environment
  env = gym.envs.make("Breakout-v0")
 


  # Create models
  model = DQN(
    K=K,
    conv_layer_sizes=conv_layer_sizes,
    hidden_layer_sizes=hidden_layer_sizes,
    gamma=gamma,
    scope="model")
  target_model = DQN(
    K=K,
    conv_layer_sizes=conv_layer_sizes,
    hidden_layer_sizes=hidden_layer_sizes,
    gamma=gamma,
    scope="target_model"
  )



  with tf.Session() as sess:
    model.set_session(sess)
    target_model.set_session(sess)
    sess.run(tf.global_variables_initializer())


    print("Populating experience replay buffer...")
    obs = env.reset()
    obs_small = downsample_image(obs)
    state = np.stack([obs_small] * 4, axis=0)
    # assert(state.shape == (4, 80, 80))
    for i in range(MIN_EXPERIENCES):

        action = np.random.choice(K)
        obs, reward, done, _ = env.step(action)
        next_state = update_state(state, obs)
        # assert(state.shape == (4, 80, 80))
        experience_replay_buffer.append((state, action, reward, next_state, done))

        if done:
            obs = env.reset()
            obs_small = downsample_image(obs)
            state = np.stack([obs_small] * 4, axis=0)
            # assert(state.shape == (4, 80, 80))
        else:
            state = next_state


    # Play a number of episodes and learn!
    for i in range(num_episodes):

      total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon = play_one(
        env,
        total_t,
        experience_replay_buffer,
        model,
        target_model,
        gamma,
        batch_sz,
        epsilon,
        epsilon_change,
        epsilon_min,
      )
      episode_rewards[i] = episode_reward

      last_100_avg = episode_rewards[max(0, i - 100):i + 1].mean()
      print("Episode:", i,
        "Duration:", duration,
        "Num steps:", num_steps_in_episode,
        "Reward:", episode_reward,
        "Training time per step:", "%.3f" % time_per_step,
        "Avg Reward (Last 100):", "%.3f" % last_100_avg,
        "Epsilon:", "%.3f" % epsilon
      )
      sys.stdout.flush()
>>>>>>> upstream/master


