def evaluate_sorting(data):
    reward = 0
    for index in range(1, len(data)):
        if data[index - 1] < data[index]:
            reward += 1
        else:
            reward -= 1
    return reward
