from project.handlers import *

class chat:
    def __init__(self):
        self.Ai = MachineCortex()
    
    def run_chat(self):
        while True:
            user_input = str(input("Voce: "))

            if user_input == "Q()":
                break

            print("IA: ", self.Ai.Dialog(user_input))
