import time
from random import random

import numpy as np
from sympy import Matrix
# from hill import HillClimbing
from ngram_score import NgramScore

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
# Tworzenie instancji Ngram_score
ngram_score = NgramScore('english_bigrams.txt')  # Zamień 'plik_z_ngramami.txt' na ścieżkę do Twojego pliku z n-gramami


def char_to_num(c):
    return alphabet.index(c)


def num_to_char(n):
    return alphabet[n]


# Funkcja do generowania losowej macierzy klucza o wymiarze n*n
def generate_random_key(n):
    while True:
        key_matrix = np.random.randint(0, 26, size=(n, n))
        det = int(np.round(np.linalg.det(key_matrix)))
        if np.gcd(det, 26) == 1:  # # Determinanta musi być 1mod26
            return key_matrix


# Funkcja szyfrowania
def hill_cipher_encrypt(plain_text, key_matrix):
    n = key_matrix.shape[0]
    cipher_text = ''
    # Podzielenie tekstu na bloki i konwersja na liczby
    blocks = [plain_text[i:i + n] for i in range(0, len(plain_text), n)]
    for block in blocks:
        block = block.upper()
        while len(block) < n:
            block += 'X'  # Dopełnianie bloków literą X
        block_vector = np.array([char_to_num(c) for c in block])

        # Mnożenie macierzy
        cipher_vector = np.dot(key_matrix, block_vector) % 26
        cipher_text += ''.join(num_to_char(num) for num in cipher_vector)

    return cipher_text


# Funkcja deszyfrowania
def hill_cipher_decrypt(cipher_text, key_matrix):
    n = key_matrix.shape[0]
    plain_text = ''

    # Inwersja macierzy klucza modulo 26
    mod = 26
    key_matrix_mod_inv = Matrix(key_matrix).inv_mod(mod)
    key_matrix_mod_inv = np.array(key_matrix_mod_inv).astype(int)

    # Podzielenie tekstu na bloki i konwersja na liczby
    blocks = [cipher_text[i:i + n] for i in range(0, len(cipher_text), n)]
    for block in blocks:
        block = block.upper()
        block_vector = np.array([char_to_num(c) for c in block])

        # Mnożenie macierzy
        plain_vector = np.dot(key_matrix_mod_inv, block_vector) % 26
        plain_text += ''.join(num_to_char(num) for num in plain_vector)

    # Usunięcie nadmiarowych 'X' na końcu tekstu
    plain_text = plain_text.rstrip('X')

    return plain_text


# Metoda Hill Climbing
import time
import numpy as np


def changeKey(old_key, max_attempts=10, num_no_improve=5):
    copy_key = old_key
    mutation_history = np.zeros_like(old_key)
    no_improve = 0
    for _ in range(max_attempts):
        new_key = old_key.copy()
        if no_improve >= num_no_improve:  # Jeśli klucz nie poprawia się przez pewien czas, zwiększamy liczbę permutacji
            num_changes = 5  # Usunąć ten fragment
        else:
            num_changes = 3
            r = random.random()
            r_probs = [0.8, 0.1, 0.1]
        for _ in range(num_changes):
            mutation_type = np.random.randint(0, key_size)
            if r < sum[r_probs[0]]:
                row, col = np.unravel_index(np.argmax(np.abs(new_key)), new_key.shape)  # Zamienić na zamianę losową
                new_key[row, col] = np.random.randint(0, 26)  # Zapewnić podział 80% / 20% szukać na teamsie
                mutation_history[row, col] += 1  # poprawić przejście tył przód
            elif r < sum[r_probs[1]]:
                row1 = np.random.randint(0, key_size)
                row2 = np.random.randint(0, key_size)
                new_key[[row1, row2]] = new_key[[row2, row1]]
            elif r < sum[r_probs[2]]:
                col1 = np.random.randint(0, key_size)
                col2 = np.random.randint(0, key_size)
                new_key[:, [col1, col2]] = new_key[:, [col2, col1]]
        det = int(np.round(np.linalg.det(new_key)))
        if np.gcd(det, 26) == 1:
            return new_key, 0  # Resetujemy licznik no_improve
        # no_improve += 1  # Zwiększamy licznik no_improve  // To też jest do wywalenia bo mówił że to i tak nie ma sensu wzgledem ngram score
    return old_key, no_improve


def hill_climbing_attack(cipher_text, Ngram_score, time_limit=30):
    old_key = generate_random_key(key_size)
    old_value = Ngram_score.score(hill_cipher_decrypt(cipher_text, old_key))
    no_improve = 0  # Dodajemy licznik no_improve
    start_time = time.time()
    while time.time() - start_time < time_limit and old_value != 0:
        new_key, no_improve = changeKey(old_key,
                                        num_no_improve=no_improve)  # Przekazujemy licznik no_improve do funkcji changeKey
        new_value = Ngram_score.score(hill_cipher_decrypt(cipher_text, new_key))
        if new_value > old_value:
            old_key = new_key
            old_value = new_value
            print(old_value, hill_cipher_decrypt(cipher_text[:3 * key_size], old_key),
                  '\n', old_key)
            no_improve = 0  # Resetujemy licznik no_improve
        else:
            no_improve += 1  # Zwiększamy licznik no_improve
        if old_value == Ngram_score.score(cipher_text):
            print("Time before breaking the key: " + str(time.time() - start_time) + "\n")
            return old_key
    print(f"Score of the best key: {old_value}")
    return old_key


if __name__ == "__main__":
    text = "No amount of evidence will ever persuade an idiot. " \
           + "When I was seventeen, my father was so stupid, " \
           + "I didnt want to be seen with him in public. " \
           + "When I was twenty four, I was amazed at how much " \
           + "the old man had learned in just seven years. " \
           + "Why waste your money looking up your family tree? " \
           + "Just go into politics and your opponent will do it for you. " \
           + "I was educated once - it took me years to get over it. " \
           + "Never argue with stupid people, they will drag you down " \
           + "to their level and then beat you with experience. " \
           + "If you don't read the newspaper, you're uninformed. " \
           + "If you read the newspaper, you're mis-informed. " \
           + "How easy it is to make people believe a lie, " \
           + "and how hard it is to undo that work again! " \
           + "Good decisions come from experience. " \
           + "Experience comes from making bad decisions. " \
           + "If you want to change the future, you must change " \
           + "what you're doing in the present. " \
           + "Don't wrestle with pigs. You both get dirty and the pig likes it. " \
           + "Worrying is like paying a debt you don't owe. " \
           + "The average woman would rather have beauty than brains, " \
           + "because the average man can see better than he can think. " \
           + "The more I learn about people, the more I like my dog."
    plain_text = ''.join([c for c in text.upper() if c in alphabet])[:300]

    start_time = time.time()
    # Generowanie losowego klucza
    key_size = 2  # Rozmiar klucza
    key_matrix = generate_random_key(key_size)

    # Szyfrowanie tekstu
    cipher_text = hill_cipher_encrypt(plain_text, key_matrix)

    # Atak Hill Climbing
    best_key = hill_climbing_attack(cipher_text, ngram_score)

    print("__________________________________")
    end_time = time.time()
    duration = end_time - start_time
    print("Duration: " + str(round(duration, 4)) + "s")
    print("__________________________________")

    print("__________________________________")
    # Wypisanie informacji
    print(f"Generated key matrix:\n{key_matrix}")
    print("__________________________________")
    print(f"Original text: {plain_text}")
    print(f"Word encrypted: {cipher_text}")

    print("__________________________________")
    print(f"Best key found during attack:\n{best_key}")
    print(f"Decrypted text with best key: {hill_cipher_decrypt(cipher_text, best_key)}")

    print("__________________________________")
    # Wypisanie macierzy do deszyfrowania
    # mod = 26
    # key_matrix_mod_inv = Matrix(key_matrix).inv_mod(mod)
    # key_matrix_mod_inv = np.array(key_matrix_mod_inv).astype(int)
    # print(f"Inverse key matrix for decryption:\n{key_matrix_mod_inv}")
    # print("__________________________________")