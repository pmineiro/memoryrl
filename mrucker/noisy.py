import random

class BernoulliNoiseLearner:
    def __init__(self, learner, noise_prob):
        self._noise_prob = noise_prob
        self._learner = learner

    def init(self):
        try:
            self._learner.init()
        except:
            pass

    @property
    def family(self):
        return self._learner.family

    @property
    def params(self):
        return {**self._learner.params, 'p': self._noise_prob}

    def choose(self, key, context, actions) -> int:
        return self._learner.choose(key,context,actions)

    def learn(self, key, context, action, reward) -> None:

        if random.random() < self._noise_prob:
            reward = abs(reward-1)

        self._learner.learn(key,context,action,reward)