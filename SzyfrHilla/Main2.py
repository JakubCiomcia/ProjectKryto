import time
import random
import numpy as np
from sympy import Matrix
from ngram_score import NgramScore

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ngram_score = NgramScore('english_bigrams.txt')


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



def changeKey(old_key):
    mutation_history = np.zeros_like(old_key)
    new_key = old_key.copy()
    num_changes = 1

    # Dynamiczne dostosowanie prawdopodobieństw mutacji
    # success_rate = np.mean(mutation_history) / np.max(mutation_history) if np.max(mutation_history) != 0 else 0
    # mutation_probabilities = [prob * (1 - adaptive_factor * success_rate) for prob in mutation_probabilities]
    # mutation_probabilities = [prob / sum(mutation_probabilities) for prob in mutation_probabilities]

    mutation_probabilities = [0.70, 0.00, 0.03, 0.24, 0.03]

    for _ in range(num_changes):
        r = random.random()
        if r < mutation_probabilities[0]:
            row, col = np.random.randint(0, new_key.shape[0]), np.random.randint(0, new_key.shape[1])
            new_key[row, col] = np.random.randint(0, 26)
            mutation_history[row, col] += 1
        elif r < sum(mutation_probabilities[:1]):
            row, col = np.random.randint(0, new_key.shape[0]), np.random.randint(0, new_key.shape[1])
            new_key[row, col] = np.random.randint(0, 26)
            mutation_history[row, col] = np.random.randint(26)
        elif r < sum(mutation_probabilities[:2]):
            row1, row2 = np.random.choice(new_key.shape[0], 2, replace=False)
            new_key[[row1, row2]] = new_key[[row2, row1]]
        # elif r < sum(mutation_probabilities[:3]):
        #     row = np.random.choice(new_key.shape[0])
        #     change = np.random.choice([-2, 2])  # Wybiera -2 lub 2 losowo
        #     new_key[row] += change
        elif r < sum(mutation_probabilities[:3]):
            row = np.random.choice(new_key.shape[0])
            changes = np.random.choice([-3, -2, -1, -1, 0,0,1,1,2,3], size=new_key.shape[1])  # Wybiera -2 lub 2 losowo dla każdego elementu w wierszu
            new_key[row] += changes
        else:
            col1, col2 = np.random.choice(new_key.shape[1], 2, replace=False)
            new_key[:, [col1, col2]] = new_key[:, [col2, col1]]

    det = int(np.round(np.linalg.det(new_key)))
    if np.gcd(det, 26) == 1:
        return new_key
    return old_key


def hill_climbing_attack(cipher_text, Ngram_score, time_limit=120, reset_limit=4000):
    old_key = generate_random_key(key_size)
    old_value = Ngram_score.score(hill_cipher_decrypt(cipher_text, old_key))
    start_time = time.time()
    attempts_since_last_improvement = 0
    best_key = old_key
    best_value = old_value

    while time.time() - start_time < time_limit:

        if attempts_since_last_improvement >= reset_limit:
            old_key = generate_random_key(key_size)
            old_value = Ngram_score.score(hill_cipher_decrypt(cipher_text, old_key))
            attempts_since_last_improvement = 0
            print("Resetting key after", reset_limit, "attempts")

        new_key = changeKey(old_key)
        new_value = Ngram_score.score(hill_cipher_decrypt(cipher_text, new_key))

        if new_value > old_value:
            old_key = new_key
            old_value = new_value
            attempts_since_last_improvement = 0
            if old_value > best_value:
                best_key = old_key
                best_value = old_value
            print(old_value, hill_cipher_decrypt(cipher_text[:3 * key_size], old_key), '\n', old_key)
        else:
            attempts_since_last_improvement += 1

        if best_value == Ngram_score.score(cipher_text):
            print("Time before breaking the key: " + str(time.time() - start_time) + "\n")
            return best_key

    print(f"Score of the best key: {best_value}")
    return best_key


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
    key_size = 3
    key_matrix = generate_random_key(key_size)
    cipher_text = hill_cipher_encrypt(plain_text, key_matrix)
    best_key = hill_climbing_attack(cipher_text, ngram_score)

    print("__________________________________")
    end_time = time.time()
    duration = end_time - start_time
    print("Duration: " + str(round(duration, 4)) + "s")
    print("__________________________________")

    print("__________________________________")
    print(f"Generated key matrix:\n{key_matrix}")
    print("__________________________________")
    print(f"Original text: {plain_text}")
    print(f"Word encrypted: {cipher_text}")

    print("__________________________________")
    print(f"Best key found during attack:\n{best_key}")
    print(f"Decrypted text with best key: {hill_cipher_decrypt(cipher_text, best_key)}")

    print("__________________________________")
