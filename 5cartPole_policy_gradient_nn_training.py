import gym
import tensorflow as tf
import numpy as np

num_inputs = 4
num_hidden = 5
num_outputs = 1

learning_rate = 0.01

initializer = tf.contrib.layers.variance_scaling_initializer()

# input to be fed to feed dict
X = tf.placeholder(tf.float32, shape=[None, num_inputs])

# takes in input
hidden_layer = tf.layers.dense(X, num_hidden, activation=tf.nn.relu, 
                kernel_initializer=initializer)

# output layer
logits = tf.layers.dense(hidden_layer, num_outputs)
outputs = tf.nn.sigmoid(logits) # for left move
# extracting the probabilities and then the action from the output layer
probabilities = tf.concat(axis=1, values=[outputs, 1-outputs])
action = tf.multinomial(probabilities, num_samples=1)

# get an output y to train the network on
y = 1.0 - tf.to_float(action) # convert tf tensor 1.0 to float number to avoid shape mismatch

# Loss function and optimizer
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)

# Instead of minimizing the optimizer, we are gonna compute the gradients on crossentropy
gradients_and_variables = optimizer.compute_gradients(cross_entropy)

gradients = []
gradient_placeholders = []
grads_and_vars_feed = []

# Separating out the gradients and variables to get gradient placeholders and variable
for gradient, variable in gradients_and_variables:
    gradients.append(gradient)
    gradient_placeholder = tf.placeholder(tf.float32, shape=gradient.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))

training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Takes in rewards and applies discount rate
def helper_discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

# Takes in all rewards, applies helper discount, then normalize them
def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = []
    for rewards in all_rewards:
        all_discounted_rewards.append(helper_discount_rewards(rewards, discount_rate))
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    # awesome one liner below
    return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]

# Environment setup and training
env = gym.make('CartPole-v0')

num_game_rounds = 10
max_game_steps = 1000
num_iterations = 650
discount_rate = 0.9

with tf.Session() as sess:
    sess.run(init)

    for iteration in range(num_iterations):
        print('On iteration: {}'.format(iteration))

        all_rewards = []
        all_gradients = []

        for game in range(num_game_rounds):
            current_rewards = []
            current_gradients = []

            observations = env.reset()

            for step in range(max_game_steps):

                action_val, gradients_val = sess.run([action, gradients], feed_dict={X:observations.reshape(1, num_inputs)})

                observations, reward, done, info = env.step(action_val[0][0])

                current_rewards.append(reward)
                current_gradients.append(gradients_val)

                if done:
                    break
            
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
        feed_dict = {}

        # apply the score we calculated to the gradients
        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            # Brace for it!!! BC!
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                        for game_index, rewards in enumerate(all_rewards)
                                            for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gradient_placeholder] = mean_gradients
        
        sess.run(training_op, feed_dict=feed_dict)

        print('Saving graph and session')

        # export graph to meta file to run this on another file (run the other file to view the rendering)
        meta_graph_def = tf.train.export_meta_graph(filename='my-policy-model.meta')
        saver.save(sess, 'my-policy-model')
        
## TODO: Install miniconda in college gpu inside docker container tensor1 and try this again
## get the trained model to laptop

env = gym.make('CartPole-v0')
observations = env.reset()

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('my-policy-model.meta')
    saver.restore(sess, 'my-policy-model')

    for x in range(500):
        env.render()
        action_val, gradients_val = sess.run([action, gradients], feed_dict={X:observations.reshape(1, num_inputs)})
        observations, reward, done, info = env.step(action_val[0][0])