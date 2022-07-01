from importlib.resources import path
import json
import os
import numpy as np
from keras.utils import np_utils
import lyricsgenius as lg



def tokenize(config: dict):
    """ Runs the preprocessing script. 
    Args: 
        config(json): config object
    Returns:
        X_vectors (np.ndarray): input vectors
        Y_vectors (np.ndarray): output vectors
        char_to_idx (dict): dictionary of characters to embedding
        idx_to_char (dict): dictionary of embedding to characters
    """
    artist_name = config.get("artist_name")
    num_songs = config.get("num_songs")
    path_to_scrape = config.get("path_to_scrape")
    output_dir = config.get("output_dir")
    context_size = config.get("context_size")


    all_lyrics_path = output_dir + artist_name + ".txt"
    if _file_exists(all_lyrics_path) and not config.get("re_process"):
        print("Found lyrics text file, loading")
        all_lyrics = _read_string_from_file(all_lyrics_path)
    else:
        artist_data = _read_file_as_json(path_to_scrape)
        print("Found {} songs".format(len(artist_data["songs"])))
        all_lyrics = _write_lyrics_to_file(artist_data, all_lyrics_path)
        print("Lyrics saved to file")
    
    print("Generating character embeddings")

    char_to_idx, idx_to_char, num_embeddings = _create_character_embeddings(all_lyrics)
    embedded_lyrics = np.array([char_to_idx[char] for char in all_lyrics])

    print("Generating train data")
    X_vectors, Y_vectors = _create_train_vectors(embedded_lyrics, num_embeddings, context_size)

    print("Implementation in progress")
    return X_vectors, Y_vectors, char_to_idx, idx_to_char


    

def _create_train_vectors(emb_list: np.ndarray, num_embeddings: int, context_size: int = 100):
    """ Creates a list of vectors for training. 
    Args: 
        emb_list: list of embeddings
        context_size: size of the context window
    """
    starting_idx = np.arange(len(emb_list) - context_size)
    context_idx = np.arange(context_size)
    perm = starting_idx.reshape(-1,1) + context_idx.reshape(1, -1)
    X_vectors = emb_list[perm] / float(num_embeddings)
    Y_vectors = emb_list[starting_idx + context_size]
    return X_vectors, np_utils.to_categorical(Y_vectors)


def _create_character_embeddings(lyrics: str) -> tuple[dict, dict, int]:
    """ Creates a character embedding for each character in the lyrics. 
    Args: 
        lyrics: lyrics to create embeddings for
    Returns:
        char_to_idx: dictionary of characters to indices
        idx_to_char: dictionary of indices to characters
        n_embeddings: number of embeddings
    """
    chars = sorted(set(lyrics))
    char_to_idx = {char: i for i, char in enumerate(chars)}
    idx_to_char = {i: char for i, char in enumerate(chars)}
    return char_to_idx, idx_to_char, len(char_to_idx)

def _read_string_from_file(file_path: str) -> str :
    """ Reads a file as a string. 
    Args: 
        file_path: path to the file
    Returns:
        str: string
    """
    with open(file_path, "r") as f:
        return f.read()    

def _read_file_as_json(path_to_file: str) -> dict :
    """ Reads a file as json. 
    Args: 
        path_to_file: path to the file
    Returns:
        dict: json object
    """
    with open(path_to_file, "r") as f:
        return json.load(f)
        
def _write_lyrics_to_file(artist, path_to_file: str):
    """ Writes the lyrics to a file. 
    Args: 
        artist: artist object
        path_to_file: path to the file
    Returns: all_lyrics (str): lyrics as one string
    """
    all_lyrics = '\n'.join(song["lyrics"] for song in artist["songs"])
    all_lyrics.replace("Embed", "")
    
    with open(path_to_file, 'w',encoding="utf-8") as file:
        file.write(all_lyrics)
    return all_lyrics
    

def _file_exists(file_path: str) -> bool :
    """ Checks if a file exists. 
    Args: 
        file_path: path to the file
    Returns:
        bool: True if file exists, False otherwise
    """
    return os.path.isfile(file_path)
