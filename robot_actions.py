from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

# Reachy Mini head range: pitch ±25°, roll ±20°, yaw ±45°
_ROLL_THINK = 15     # degrees — tilt right while processing
_PITCH_CONFUSED = 10  # degrees — slight downward droop


class LancerRobot:
    def __init__(self):
        try:
            self.mini = ReachyMini()
        except Exception as e:
            print(f"[LancerRobot] Failed to connect: {e}")
            self.mini = None

    def thinking(self):
        if not self.mini:
            return
        pose = create_head_pose(roll=_ROLL_THINK, degrees=True)
        self.mini.goto_target(head=pose, duration=0.5)

    def answering(self):
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
        if not self.mini:
            return
        self.mini.antennas.sad()
        pose = create_head_pose(pitch=_PITCH_CONFUSED, degrees=True)
        self.mini.goto_target(head=pose, duration=0.5)

    def speak(self, text: str):
        """Blocking — called via asyncio.to_thread so it serializes through _speak_lock."""
        if not self.mini:
            return
        self.mini.say(text)
