import time

import numpy as np
from sympy import Matrix

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def char_to_num(c):
    return alphabet.index(c)

def num_to_char(n):
    return alphabet[n]

# Funkcja do generowania losowej macierzy klucza o wymiarze n*n
def generate_random_key(n):
    while True:
        key_matrix = np.random.randint(0, 26, size=(n, n))
        det = int(np.round(np.linalg.det(key_matrix)))
        if np.gcd(det, 26) == 1:  # Determinant must be coprime with 26
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
def hill_climbing_attack(cipher_text, iterations=10000000):
    best_key = None
    best_score = 0
    i = 0
    while i < iterations:  # Pętla będzie działać dopóki i < iterations
        i += 1
        # print(i)
        # Generowanie losowego klucza
        key_size = 4  # Rozmiar klucza
        key_matrix = generate_random_key(key_size)
        # Deszyfrowanie z użyciem bieżącego klucza
        decrypted_text = hill_cipher_decrypt(cipher_text, key_matrix)
        # Obliczanie oceny jako ilość poprawnie odszyfrowanych liter
        score = sum(1 for a, b in zip(plain_text, decrypted_text) if a == b)

        # Aktualizacja najlepszego klucza i wyniku
        if score > best_score:
            best_key = key_matrix
            best_score = score
        if best_score == len(cipher_text):
            print("Ilość iteracji: " + str(i))
            return best_key
            break
    return best_key


if __name__ == "__main__":
    plain_text = "HOHO"

    start_time = time.time()
    # Generowanie losowego klucza
    key_size = 4  # Rozmiar klucza
    key_matrix = generate_random_key(key_size)

    # Szyfrowanie tekstu
    cipher_text = hill_cipher_encrypt(plain_text, key_matrix)

    # Atak Hill Climbing
    best_key = hill_climbing_attack(cipher_text)

    # Wypisanie informacji
    print(f"Original text: {plain_text}")
    print(f"Word encrypted: {cipher_text}")

    print("__________________________________")
    print(f"Best key found during attack:\n{best_key}")
    print("__________________________________")

    print(f"Decrypted text with best key: {hill_cipher_decrypt(cipher_text, best_key)}")
    print(f"Generated key matrix:\n{key_matrix}")
    # Wypisanie macierzy do deszyfrowania
    mod = 26
    key_matrix_mod_inv = Matrix(key_matrix).inv_mod(mod)
    key_matrix_mod_inv = np.array(key_matrix_mod_inv).astype(int)
    print(f"Inverse key matrix for decryption:\n{key_matrix_mod_inv}")
    end_time = time.time()
    duration = end_time - start_time
    print("Czas trwania: " + str(round(duration, 4)))