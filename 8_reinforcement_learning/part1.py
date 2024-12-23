import random
import gymnasium as gym
import numpy as np

num_player_sums = 32  # Hvorfor er denne 32? 21 + 11? hvis man sticker på en 21?
num_dealer_cards = 11  # dealer-kortet kan ha verdi 1(ess) til 11(ess)
num_usable_ace = 2  # boolean for om man har en tilgjengelig usable ace
num_actions = 2  # stick eller hit

num_episodes = 10000
epsilon = 0.20
epsilon_min = 0.10
epsilon_decay = (epsilon - epsilon_min) / (
        num_episodes / 2)  # Ferdig med decay halvveis

gamma = 0.1

showHuman = 0

env = gym.make("Blackjack-v1", render_mode="human" if showHuman else None)
loss_draw_win = [0, 0, 0]

Q = np.zeros((num_player_sums, num_dealer_cards, num_usable_ace, num_actions))

for episode in range(num_episodes + 1):
    observation, info = env.reset() # seed 0 gjør at dealer har knekt hver gang?

    while True:
        (player_sum, dealer_card, usable_ace) = observation

        action = None
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            # print(Q[player_sum, dealer_card, usable_ace])
            action = np.argmax(Q[player_sum, dealer_card, usable_ace])

        (next_observation, reward, terminated, truncated, _) = env.step(action)

        next_player_sum, next_dealer_card, next_usable_ace = next_observation

        Q[player_sum, dealer_card, usable_ace, action] += \
            epsilon * (
                reward +
                gamma * np.max( #Finner beste fremtidige reward?
                Q[next_player_sum, next_dealer_card, next_usable_ace]) -
                Q[player_sum, dealer_card, usable_ace, action]
            )
        observation = next_observation

        if terminated:
            loss_draw_win[int(reward) + 1] += 1
            break
        elif truncated:
            print("Truncated")
            break

    epsilon = max(epsilon - epsilon_decay, epsilon_min)

    # print(f"epsilon is {epsilon}")
    if episode % 1000 == 0:
        print(f"\nStatus at episode {episode}")
        total = sum(loss_draw_win)
        print(
            f"Losses: {loss_draw_win[0]}, {(loss_draw_win[0] / total) * 100:.2f}%")
        print(
            f"Draws: {loss_draw_win[1]}, {(loss_draw_win[1] / total) * 100:.2f}%")
        print(
            f"Wins: {loss_draw_win[2]}, {(loss_draw_win[2] / total) * 100:.2f}%")
        loss_draw_win = [0, 0, 0]
env.close()
