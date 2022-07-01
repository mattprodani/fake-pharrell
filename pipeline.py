import datetime
import json
from scraper import save_lyrics_to_file
import os
from preprocess import tokenize
from train import train_model

config = json.load(open("config.json"))
artist_name = config.get("artist_name")

def main():
    """ Main function. """

    # Read the config file
    path_to_scrape = config.get("path_to_scrape")

    # Check if the file exists
    if os.path.isfile(path_to_scrape):
        print("Found artist file")
    else:
        if config.get("do_not_scrape"): raise FileNotFoundError
        save_lyrics_to_file(config)

    print("Processing lyrics")
    X_vectors, Y_vectors, char_to_idx, idx_to_char = tokenize(config)
    print("Lyrics processed")
    
    print("Training model")
    # Train the model
    model = train_model(X_vectors, Y_vectors, config)
    print("Model trained")
    # Save the model
    save_model_data(model, char_to_idx, idx_to_char)
    print("Model saved")
    return 1, model, char_to_idx, idx_to_char


def save_model_data(model, char_to_idx, idx_to_char):
    """ Saves the model to a file. """
    suffix = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    model.save(f"models/{artist_name}_{suffix}c.h5")
    model.save_weights(f"models/{artist_name}_{suffix}c_weights.h5")
    with open("embeddings/char_to_idx.json", "w") as f:
        json.dump(char_to_idx, f)
    with open("embeddings/idx_to_char.json", "w") as f:
        json.dump(idx_to_char, f)

    print("Model saved")


def __main__():
    if __name__ == "__main__":
        main()
    pass
