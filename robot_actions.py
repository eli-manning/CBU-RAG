from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


class LancerRobot:
    def __init__(self):
        try:
            self.mini = ReachyMini()
        except Exception as e:
            print(f"[LancerRobot] Failed to connect: {e}")
            self.mini = None

    def thinking(self):
        """Head tilt while processing"""
        if not self.mini:
            return
        pose = create_head_pose(roll=10, degrees=True)
        self.mini.goto_target(head=pose, duration=0.5)

    def answering(self):
        """Look forward when speaking"""
        if not self.mini:
            return
        pose = create_head_pose()  # neutral
        self.mini.goto_target(head=pose, duration=0.4)

    def greet(self):
        if not self.mini:
            return
        self.mini.antennas.happy()
        self.mini.say("Hey! I'm Lancer, CBU's ACM chatbot.")

    def confused(self):
        """When RAG returns no good context"""
        if not self.mini:
            return
        self.mini.antennas.sad()
        pose = create_head_pose(pitch=-5, degrees=True)
        self.mini.goto_target(head=pose, duration=0.5)

    def speak(self, text: str):
        if not self.mini:
            return
        self.mini.say(text)
