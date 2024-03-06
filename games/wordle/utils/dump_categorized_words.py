import clemgame
from clemgame import file_utils

logger = clemgame.get_logger(__name__)


def read_file_contents(filename, file_ext="txt"):
    if file_ext == "csv":
        words_dict = {}
        words_list = file_utils.load_csv(f"resources/{filename}", "wordle")
        if filename == "nytcrosswords.csv":
            for word in words_list:
                words_dict[word[1].lower().strip()] = word[2].lower().strip()
        elif filename == "unigram_freq.csv":
            words_list = words_list[1:]
            for word, freq in words_list:
                words_dict[word.lower().strip()] = freq
        else:
            return words_list
        return words_dict

    elif file_ext == "txt":
        words = file_utils.load_file(f"resources/{filename}", "wordle")
        words_list = words.split("\n")
        words_list = [word.lower().strip() for word in words_list]
        return words_list


def get_freq(word_list, freq_dict, clue_dict):
    """
    Returns a dictionary of words and their frequencies
    Checks if the target word is present in the frequency dictionary and the clue dictionary
    """
    data = {}
    for word in word_list:
        if word in freq_dict and word in clue_dict:
            data[word] = freq_dict[word]
    return data


def write_to_file(data, filename):
    # Write the data to a file
    with open(filename, "w") as fp:
        fp.write("\n".join(data))

# Function to classify each frequency
def classify_frequency(freq, mean_freq, std_dev_freq):
    if freq > mean_freq + std_dev_freq:
        return 'High'
    elif freq < mean_freq - std_dev_freq:
        return 'Low'
    else:
        return 'Medium'


def start_word_categorization():
    unigram_freq_file = "unigram_freq.csv"
    all_possible_wordle_words_files = "wordle_recognized_english_words.txt"
    target_words_file = "wordle_target_words.txt"
    clue_file = "nytcrosswords.csv"

    unigram_freq = read_file_contents(unigram_freq_file, file_ext="csv")
    # all_wordle_words = read_file_contents(all_possible_wordle_words_files)
    target_words = read_file_contents(target_words_file)
    clue_words = read_file_contents(clue_file, file_ext="csv")

    sorted_unigram_freq = sorted(unigram_freq.items(), key=lambda x: int(x[1]), reverse=True)
    print(f"Unigram Frequency:: Min = {sorted_unigram_freq[-1][1]}, Max = {sorted_unigram_freq[0][1]}, Median = {sorted_unigram_freq[int(len(sorted_unigram_freq)/2)][1]}")

    target_words_freq_dict = get_freq(target_words, unigram_freq, clue_words)
    # Sort the dictionary by value
    sorted_dict = sorted(target_words_freq_dict.items(), key=lambda x: int(x[1]), reverse=True)

    # Get the word frequencies
    target_word_frequencies = [int(word[1]) for word in sorted_dict]
    mean_freq = np.mean(target_word_frequencies)
    std_dev_freq = np.std(target_word_frequencies)

    # Classify the frequencies
    """
    High: Frequency > mean + std_dev
    Low: Frequency < mean - std_dev
    Medium: mean - std_dev < Frequency < mean + std_dev

    This classification is based on the assumption that the frequency distribution is normal
    However most of the words are in the medium category and no words are in the low category

    Since clemgame analysis is not based on the frequency categorization, keeping the classification as is

    For future work:
    1. The classification can be improved by using a different method to classify the frequencies
    2. Use a different frequency distribution file
    3. The word may be complex depending on the letters used in the word [complex word with simple letters, simple word with complex letters, etc.]
    """
    easy_words, medium_words, hard_words = [], [], []
    for word, freq in sorted_dict:
        freq = int(freq)
        if classify_frequency(freq, mean_freq, std_dev_freq) == 'High':
            easy_words.append((word, freq))
        elif classify_frequency(freq, mean_freq, std_dev_freq) == 'Low':
            hard_words.append((word, freq))
        else:
            medium_words.append((word, freq))

    print(f"Easy Words:: Length = {len(easy_words)}")
    print(f"Medium Words:: Length = {len(medium_words)}")
    print(f"Hard Words:: Length = {len(hard_words)}")

    easy_words = [word[0] for word in easy_words]
    medium_words = [word[0] for word in medium_words]
    hard_words = [word[0] for word in hard_words]

    write_to_file(
        easy_words,
        "games/wordle/resources/easy_words.txt",
    )
    write_to_file(
        medium_words,
        "games/wordle/resources/medium_words.txt",
    )
    write_to_file(
        hard_words,
        "games/wordle/resources/hard_words.txt",
    )



if __name__ == "__main__":
    start_word_categorization()
