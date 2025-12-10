
class Episode:

    episodes = []

    def __init__(self, run):
        self.run = run
        self.transitions = []
        self.steps = 0
        Episode.episodes.append(self)
        self.number = len(Episode.episodes)

    def AddTransition(self, t):
        self.transitions.append(t)
        self.steps += 1

    def GetScore(self):

        score = 0
        for t in self.transitions:
            score += t.reward
        return score
    
    def GetSteps(self):
        return len(self.transitions)
    
    def GetStep(i):
        return self.transitions[i]
    
    @staticmethod
    def GetEpisode(i):
        return Episode.episodes[i]
    
    @staticmethod
    def GetLength():
        return len(Episode.episodes)