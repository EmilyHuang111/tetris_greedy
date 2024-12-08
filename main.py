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
        print("Training completed.")
    else:
        while True:
            if mode == "student":
                # Load the trained AI model and run the game
                try:
                    ai_model = CUSTOM_AI_MODEL(weights_path='trained_model.npy')
                except FileNotFoundError:
                    print("Trained model not found. Please train the model first.")
                    sys.exit(1)
                game = Game(mode='student', agent=ai_model)
            else:
                # Handle other modes or run with no AI
                game = Game(mode=mode)
            
            # Run the game
            game.run()
            
            # After the game finishes, ask the user if they want to play again
            while True:
                choice = input("Do you want to play again? (y/n): ").strip().lower()
                if choice in ['y', 'yes']:
                    break  # Restart the loop to play again
                elif choice in ['n', 'no']:
                    print("Exiting the game. Goodbye!")
                    sys.exit(0)
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")

if __name__ == "__main__":
    main()
