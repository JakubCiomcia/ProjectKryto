import time
import numpy as np
from sympy import Matrix

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
char_to_num_cache = {c: i for i, c in enumerate(alphabet)}
num_to_char_cache = {i: c for i, c in enumerate(alphabet)}

def char_to_num(c):
    return char_to_num_cache[c]

def num_to_char(n):
    return num_to_char_cache[n]

def generate_random_key(n):
    while True:
        key_matrix = np.random.randint(0, 26, size=(n, n))
        det = np.linalg.det(key_matrix)
        if np.gcd(int(round(det)), 26) == 1:
            return key_matrix

def hill_cipher_encrypt(plain_text, key_matrix):
    n = key_matrix.shape[0]
    block_size = n
    plain_text = plain_text.upper().replace(" ", "")
    padding = block_size - len(plain_text) % block_size
    plain_text += 'X' * padding
    plain_text_matrix = np.array([char_to_num(c) for c in plain_text])
    plain_text_matrix = plain_text_matrix.reshape(-1, block_size)
    cipher_matrix = np.dot(plain_text_matrix, key_matrix) % 26
    cipher_text = ''.join([num_to_char(num) for row in cipher_matrix for num in row])
    return cipher_text

def hill_cipher_decrypt(cipher_text, key_matrix):
    n = key_matrix.shape[0]
    block_size = n
    cipher_text_matrix = np.array([char_to_num(c) for c in cipher_text])
    cipher_text_matrix = cipher_text_matrix.reshape(-1, block_size)
    inverse_key_matrix = Matrix(key_matrix).inv_mod(26)
    inverse_key_matrix = np.array(inverse_key_matrix).astype(int)
    plain_text_matrix = np.dot(cipher_text_matrix, inverse_key_matrix) % 26
    plain_text = ''.join([num_to_char(num) for row in plain_text_matrix for num in row])
    return plain_text.rstrip('X')

def hill_climbing_attack(cipher_text, iterations=10000000):
    best_key = None
    best_score = 0
    i = 0
    plain_text = "HOHO"  # Pre-compute the plain_text
    while i < iterations:
        i += 1
        key_matrix = generate_random_key(3)
        decrypted_text = hill_cipher_decrypt(cipher_text, key_matrix)
        score = sum(1 for a, b in zip(plain_text, decrypted_text) if a == b)
        if score > best_score:
            best_key = key_matrix
            best_score = score
        if best_score == len(cipher_text):
            print("Ilość iteracji: " + str(i))
            return best_key
            break
    return best_key

if __name__ == "__main__":
    plain_text = "HOH"
    start_time = time.time()
    key_matrix = generate_random_key(3)
    cipher_text = hill_cipher_encrypt(plain_text, key_matrix)
    best_key = hill_climbing_attack(cipher_text)
    print(f"Original text: {plain_text}")
    print(f"Word encrypted: {cipher_text}")
    print("__________________________________")
    print(f"Best key found during attack:\n{best_key}")
    print("__________________________________")
    print(f"Decrypted text with best key: {hill_cipher_decrypt(cipher_text, best_key)}")
    print(f"Generated key matrix:\n{key_matrix}")
    inverse_key_matrix = Matrix(key_matrix).inv_mod(26)
    inverse_key_matrix = np.array(inverse_key_matrix).astype(int)
    print(f"Inverse key matrix for decryption:\n{inverse_key_matrix}")
    end_time = time.time()
    duration = end_time - start_time
    print("Czas trwania: " + str(round(duration, 4)))
