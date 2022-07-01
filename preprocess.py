from importlib.resources import path
import json
import os
from scraper import save_lyrics_to_file
import numpy as np
from keras.utils import np_utils

scrape_dir = "scrapes/"
process_dir = "processed/"


def run(path_to_file, artist_name, num_songs, dir = "scrapes/", file_name = None):
    """ Runs the preprocessing script. 
    Args: 
        filename: path to the file to save the lyrics to
        artist_name: name of the artist
        num_songs: number of songs to return
        dir: dir to save the file to (include / at the end) Default: scrapes/
        file_name: name of the file to save the lyrics, leave blank to use artist name
    """
    

    if _file_exists(path_to_file):
        print("Found artist file")
    else:
        print("Did not find artist file, processing lyrics from API")
        save_lyrics_to_file(artist_name, num_songs, file_name=path_to_file)

    artist_data = _read_file_as_json(path_to_file)
    print("Found {} songs".format(len(artist_data["songs"])))


    print("Saving lyrics to file")
    all_lyrics_path = process_dir + artist_name + ".txt"
    if _file_exists(all_lyrics_path):
        print("Found lyrics text file, loading")
        all_lyrics = _read_string_from_file(all_lyrics_path)
    else:
        all_lyrics = _write_lyrics_to_file(artist_data, all_lyrics_path)
        print("Lyrics saved to file")
    
    print("Generating character embeddings")

    char_to_idx, idx_to_char, num_embeddings = _create_character_embeddings(all_lyrics)
    embedded_lyrics = [char_to_idx[char] for char in all_lyrics]

    print("Generating train data")
    X_vectors, Y_vectors = _create_train_vectors(embedded_lyrics, num_embeddings, context_size = 100)


    

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
    chars = sorted(set(str))
    char_to_idx = {char: i for i, char in enumerate(chars)}
    idx_to_char = {i: char for i, char in enumerate(chars)}
    return char_to_idx, idx_to_char, len(char_to_idx)

def _read_string_from_file(file_path: str) -> str:
    """ Reads a file as a string. 
    Args: 
        file_path: path to the file
    Returns:
        str: string
    """
    with open(file_path, "r") as f:
        return f.read()    

def _read_file_as_json(path_to_file: str) -> dict:
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
    all_lyrics = '\n'.join(song.lyrics for song in artist.songs)
    with open(path_to_file, 'w',encoding="utf-8") as file:
        file.write(all_lyrics)
    return all_lyrics
    

def _file_exists(file_path):
    """ Checks if a file exists. 
    Args: 
        file_path: path to the file
    Returns:
        bool: True if file exists, False otherwise
    """
    return os.path.isfile(file_path)
