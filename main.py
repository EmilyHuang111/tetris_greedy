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
        sys.exit(0)  # Exit after training
    else:
        # For modes that require continuous gameplay
        while True:
            try:
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
                
                # Optionally, add a short pause before restarting
                # import time
                # time.sleep(1)  # Pause for 1 second
                
            except KeyboardInterrupt:
                print("\nGame interrupted by user. Exiting gracefully.")
                sys.exit(0)
            except Exception as e:
                print(f"An error occurred: {e}")
                sys.exit(1)

if __name__ == "__main__":
    main()
