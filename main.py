# main.py

import sys
from game import Game
from custom_model import CUSTOM_AI_MODEL

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [mode]")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "train":
        # Import and run train.py's training function
        import train
        train.run_X_epochs()
    elif mode == "student":
        # Load the trained AI model and run the game
        ai_model = CUSTOM_AI_MODEL(weights_path='trained_model.npy')
        game = Game(mode='student', agent=ai_model)
        game.run()
    else:
        # Handle other modes or run with no AI
        game = Game(mode=mode)
        game.run()

if __name__ == "__main__":
    main()
