import numpy as np

# Funkcja do generowania macierzy klucza
def generate_key_matrix(key):
    n = int(len(key) ** 0.5)
    key_matrix = np.array([ord(char) - ord('A') for char in key]).reshape(n, n)
    return key_matrix

# Funkcja do generowania macierzy dla tekstu do zaszyfrowania
def generate_text_matrix(text, n):
    text = text.upper().replace(" ", "").replace("\n", "")
    # Wypełniamy brakujące miejsca tekstu zerami
    while len(text) % n != 0:
        text += 'Z'
    text_matrix = np.array([ord(char) - ord('A') for char in text]).reshape(-1, n)
    return text_matrix

# Funkcja szyfrująca
def encrypt(text, key):
    n = int(len(key) ** 0.5)
    key_matrix = generate_key_matrix(key)
    text_matrix = generate_text_matrix(text, n)
    encrypted_text = ""
    for chunk in text_matrix:
        encrypted_chunk = np.dot(chunk, key_matrix) % 26
        encrypted_text += ''.join([chr(char + ord('A')) for char in encrypted_chunk])
    return encrypted_text

# Funkcja deszyfrująca
def decrypt(text, key):
    n = int(len(key) ** 0.5)
    key_matrix = generate_key_matrix(key)
    # Obliczamy odwrotność macierzy klucza modulo 26
    det = int(round(np.linalg.det(key_matrix)))
    det_inv = pow(det, -1, 26)
    key_matrix_inv = (det_inv * np.round(det * np.linalg.inv(key_matrix)).astype(int)) % 26
    text_matrix = generate_text_matrix(text, n)
    decrypted_text = ""
    for chunk in text_matrix:
        decrypted_chunk = np.dot(chunk, key_matrix_inv) % 26
        decrypted_text += ''.join([chr(char + ord('A')) for char in decrypted_chunk])
    return decrypted_text

def generate_random_key(key_size):
    # Sprawdzenie, czy rozmiar klucza jest kwadratem liczby całkowitej
    if not np.sqrt(key_size).is_integer():
        raise ValueError("Rozmiar klucza musi być kwadratem liczby całkowitej.")

    # Generowanie losowej macierzy kwadratowej o rozmiarze key_size
    random_matrix = np.random.randint(0, 26, size=(key_size, key_size))

    # Sprawdzenie, czy macierz jest odwracalna modulo 26
    det = int(np.round(np.linalg.det(random_matrix)))
    det_inv = pow(det, -1, 26)  # Odwrócenie wyznacznika modulo 26
    if det == 0 or np.gcd(det, 26) != 1:
        raise ValueError("Nie można wygenerować klucza, ponieważ macierz nie jest odwracalna modulo 26.")

    # Zwracanie wygenerowanego losowego klucza
    return random_matrix

# Przykładowe użycie:
try:
    key_size = 3  # Rozmiar klucza (np. dla szyfru Hilla 3x3)
    key = generate_random_key(key_size)
    print("Wygenerowany losowy klucz:")
    print(key)
except ValueError as e:
    print(e)
# Przykładowe użycie
key = "GYBNQKURP"
plaintext = "HELLOO"
encrypted_text = encrypt(plaintext, key)
print("Zaszyfrowany tekst:", encrypted_text)
decrypted_text = decrypt(encrypted_text, key)

print("Odszyfrowany tekst:", decrypted_text)

