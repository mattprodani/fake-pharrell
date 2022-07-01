import lyricsgenius

TOKEN = "tEfu4OW1rgHlo8c7Ab7MSiKnniPaP8bGwaPQLKrEl6U7oKhTS9on2kr85phflwJC"
genius = lyricsgenius.Genius(TOKEN, skip_non_songs = True, retries = 5)


def save_lyrics_to_file(config: dict) -> str:
    """ Saves the lyrics of a list of songs by an artist to a file. 
    Args: 
        artist_name: name of the artist
        num_songs: number of songs to return
        file_name: path to lyrics file, leave blank to use artist name
    Returns:
        str: path to the file
    """

    artist_name = config.get("artist_name")
    num_songs = config.get("num_songs")
    path_to_scrape = config.get("path_to_scrape")
    try:
        artist = genius.search_artist(artist_name, max_songs=num_songs)
    except Exception as e:
        print(e)
        exit()
    file_name = file_name if file_name else artist_name
    artist.save_lyrics(filename=path_to_scrape, sanitize=False, overwrite=True)

    return file_name



