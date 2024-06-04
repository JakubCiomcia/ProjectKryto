import time
import numpy as np
from sympy import Matrix

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def char_to_num(c):
    return alphabet.index(c)

def num_to_char(n):
    return alphabet[n]

def generate_random_key(n):
    while True:
        key_matrix = np.random.randint(0, 26, size=(n, n))
        det = int(np.round(np.linalg.det(key_matrix)))
        if np.gcd(det, 26) == 1:
            return key_matrix

def hill_cipher_encrypt(plain_text, key_matrix):
    n = key_matrix.shape[0]
    cipher_text = ''
    blocks = [plain_text[i:i + n] for i in range(0, len(plain_text), n)]
    for block in blocks:
        block = block.upper()
        while len(block) < n:
            block += 'X'
        block_vector = np.array([char_to_num(c) for c in block])
        cipher_vector = np.dot(key_matrix, block_vector) % 26
        cipher_text += ''.join(num_to_char(num) for num in cipher_vector)
    return cipher_text

def hill_cipher_decrypt(cipher_text, key_matrix):
    n = key_matrix.shape[0]
    plain_text = ''
    mod = 26
    key_matrix_mod_inv = Matrix(key_matrix).inv_mod(mod)
    key_matrix_mod_inv = np.array(key_matrix_mod_inv).astype(int)
    blocks = [cipher_text[i:i + n] for i in range(0, len(cipher_text), n)]
    for block in blocks:
        block = block.upper()
        block_vector = np.array([char_to_num(c) for c in block])
        plain_vector = np.dot(key_matrix_mod_inv, block_vector) % 26
        plain_text += ''.join(num_to_char(num) for num in plain_vector)
    plain_text = plain_text.rstrip('X')
    return plain_text

def frequency_analysis_attack(cipher_text):
    best_key = None
    best_score = 0
    max_iterations = 10000000
    i = 0
    while i < max_iterations:
        i += 1
        key_size = 3
        key_matrix = generate_random_key(key_size)
        decrypted_text = hill_cipher_decrypt(cipher_text, key_matrix)
        score = calculate_score(decrypted_text)
        if score > best_score:
            best_key = key_matrix
            best_score = score
        if best_score == len(cipher_text):
            print("Number of iterations:", i)
            return best_key
            break
    return best_key

def calculate_score(decrypted_text):
    expected_frequencies = {
        'E': 12.02, 'T': 9.10, 'A': 8.12, 'O': 7.68, 'I': 7.31,
        'N': 6.95, 'S': 6.28, 'R': 6.02, 'H': 5.92, 'D': 4.32,
        'L': 3.98, 'U': 2.88, 'C': 2.71, 'M': 2.61, 'F': 2.30,
        'Y': 2.11, 'W': 2.09, 'G': 2.03, 'P': 1.82, 'B': 1.49,
        'V': 1.11, 'K': 0.69, 'X': 0.17, 'Q': 0.11, 'J': 0.10,
        'Z': 0.07
    }
    letter_counts = {letter: decrypted_text.count(letter) for letter in alphabet}
    score = sum((letter_counts.get(letter, 0) - expected_frequencies[letter]) ** 2 for letter in alphabet)
    return score

if __name__ == "__main__":
    plain_text = "ABW"

    start_time = time.time()
    key_size = 3
    key_matrix = generate_random_key(key_size)
    cipher_text = hill_cipher_encrypt(plain_text, key_matrix)

    best_key = frequency_analysis_attack(cipher_text)

    print(f"Original text: {plain_text}")
    print(f"Word encrypted: {cipher_text}")

    print("__________________________________")
    print(f"Best key found during attack:\n{best_key}")
    print("__________________________________")

    print(f"Decrypted text with best key: {hill_cipher_decrypt(cipher_text, best_key)}")
    print(f"Generated key matrix:\n{key_matrix}")
    mod = 26
    key_matrix_mod_inv = Matrix(key_matrix).inv_mod(mod)
    key_matrix_mod_inv = np.array(key_matrix_mod_inv).astype(int)
    print(f"Inverse key matrix for decryption:\n{key_matrix_mod_inv}")
    end_time = time.time()
    duration = end_time - start_time
    print("Duration:", round(duration, 4))
