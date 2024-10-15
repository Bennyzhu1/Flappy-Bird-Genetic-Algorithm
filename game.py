import flappy_bird_gymnasium
import gymnasium
env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

obs, _ = env.reset()

"""
Thoughts
- We have a initial observation given to our neural network. Things like the bird's vertical position, the birds velocity, the pipe's position, the middle of the gap of the pipe (i feel like that is all that is needed)
- put that into some hidden layer that does some calculations
- output is a single value that is the jump or no jump

- we initialize a population of birds, say around 30-50, and we run them all and see which ones are "the most fit". We keep those that are, and then start to cross them to create offspring
- we then mutate the offspring to create a new population

- we keep doing this until we have a bird that can play until a specified score (say 1000)
"""
# Bird will be the neural network of the bird we want, returns the reward of the bird
def fitness_of_bird(bird):
    total_reward = 0
    while True:
        # Next action:
        # (feed the observation to your agent here)
        # obs
        action = env.action_space.sample()

        # Processing:
        obs, reward, terminated, _, info = env.step(action)
        total_reward += reward
        print(obs)

        # Checking if the player is still alive
        if terminated:
            break

    env.close()
    return total_reward

# have some other function here that will order the rewards from greatest to least and then then choose the top n birds to breed
def genetic_algorithm():
    pass