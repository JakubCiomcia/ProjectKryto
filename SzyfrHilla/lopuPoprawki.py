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
def hill_climbing_attack(cipher_text, iterations=10000000):
    best_key = None
    best_score = 0
    i = 0
    while i < iterations:  # Pętla będzie działać dopóki i < iterations
        i += 1
        # print(i)
        # Generowanie losowego klucza
        key_size = 3  # Rozmiar klucza
        key_matrix = generate_random_key(key_size)
        # Deszyfrowanie z użyciem bieżącego klucza
        decrypted_text = hill_cipher_decrypt(cipher_text, key_matrix)
        # Obliczanie oceny jako ilość poprawnie odszyfrowanych liter
        score = sum(1 for a, b in zip(plain_text, decrypted_text) if a == b)

        # Aktualizacja najlepszego klucza i wyniku
        if score > best_score:
            best_key = key_matrix
            best_score = score
            print( best_score, hill_cipher_decrypt(cipher_text[:27], key_matrix),
                   '\n', best_key )
        if best_score == len(cipher_text):
            print("Number of iterations before breaking the key: " + str(i) + "\n")
            return best_key
            break
    return best_key


if __name__ == "__main__":
    plain_text = "ABW"

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
    plain_text = ''.join([ c for c in text.upper() if c in alphabet ])[:300]
    
    start_time = time.time()
    # Generowanie losowego klucza
    key_size = 3  # Rozmiar klucza
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
    print(f"Inverse key matrix for decryption:\n{key_matrix_mod_inv}\n")
    end_time = time.time()
    duration = end_time - start_time
    print("Duration: " + str(round(duration, 4)) + "s")
