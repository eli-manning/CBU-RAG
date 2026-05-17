from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


class LancerRobot:
    def __init__(self):
        self.mini = ReachyMini()

    def thinking(self):
        """Head tilt while processing"""
        pose = create_head_pose(roll=10, degrees=True)
        self.mini.goto_target(head=pose, duration=0.5)

    def answering(self):
        """Look forward when speaking"""
        pose = create_head_pose()  # neutral
        self.mini.goto_target(head=pose, duration=0.4)

    def greet(self):
        self.mini.antennas.happy()
        self.mini.say("Hey! I'm Lancer, CBU's ACM chatbot.")

    def confused(self):
        """When RAG returns no good context"""
        self.mini.antennas.sad()
        pose = create_head_pose(pitch=-5, degrees=True)
        self.mini.goto_target(head=pose, duration=0.5)

    def speak(self, text: str):
        self.mini.say(text)
