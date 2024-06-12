import math
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
        elif r < sum(mutation_probabilities[:3]):
            row = np.random.choice(new_key.shape[0])
            changes = np.random.choice([-3, -2, -1, -1, 0, 0, 1, 1, 2, 3], size=new_key.shape[1])
            new_key[row] += changes
        else:
            col1, col2 = np.random.choice(new_key.shape[1], 2, replace=False)
            new_key[:, [col1, col2]] = new_key[:, [col2, col1]]

    det = int(np.round(np.linalg.det(new_key)))
    if np.gcd(det, 26) == 1:
        return new_key
    return old_key

def AcceptanceFunction(valueOld, valueNew, temp, multiplier, worse_prob):
    if random.random() < math.exp(multiplier * (valueOld - valueNew) / temp):
        return True
    elif random.random() < worse_prob:
        return True
    else:
        return False

def SimAnnealing_self_adjusting(ct, lenk, time_limit=120, tempDeltaBase=-0.001, acceptance_multiplier=-3, max_attempts=1000, worse_accept_prob_start=0.6, worse_accept_prob_end=0.3):
    t1 = time.time()
    starttemp = 100
    endtemp = 1
    temp = starttemp
    tempDelta = tempDeltaBase

    keyOld = generate_random_key(lenk)
    scoreOld = ngram_score.score(hill_cipher_decrypt(ct, keyOld))
    keyMax, scoreMax = list(keyOld), float(scoreOld)

    ctScore, iters = ngram_score.score(ct), 0

    def distance2solution(currScore, lenText=len(ct), ctScore=ctScore):
        return (-2.35 - currScore / lenText) / (-2.35 - ctScore / lenText)

    def progressMarker(currScore):
        pm = (temp / starttemp) - distance2solution(currScore)
        return pm

    j, k, j_list, k_list, restarts = 0, 0, [], [], max(1, lenk - 10)
    m = 0
    consecutive_attempts = 0

    def sign(a):
        return bool(a > 0) - bool(a < 0)

    while temp >= endtemp and (time.time() - t1) < time_limit:
        keyNew = changeKey(keyOld)
        scoreNew = ngram_score.score(hill_cipher_decrypt(ct, keyNew))
        worse_prob = worse_accept_prob_start + (worse_accept_prob_end - worse_accept_prob_start) * (temp - endtemp) / (starttemp - endtemp)

        if scoreNew > scoreOld:
            keyOld, scoreOld = keyNew, scoreNew
            k += 1
            consecutive_attempts = 0
            if scoreOld > scoreMax:
                j_list.append(j)
                k_list.append(k)
                print(f'{scoreOld},\t{round(time.time() - t1, 2)} sec,\t temp = {round(temp, 2)},\t j={j}, k={k}, \t{len(keyOld)}')
                keyMax, scoreMax, j, k = keyOld, scoreOld, 0, 0
        elif AcceptanceFunction(scoreOld, scoreNew, temp, acceptance_multiplier, worse_prob):
            keyOld, scoreOld = keyNew, scoreNew
            k += 1
            consecutive_attempts = 0
        else:
            consecutive_attempts += 1

        if consecutive_attempts > max_attempts:
            keyOld, scoreOld, consecutive_attempts = keyMax, scoreMax, 0

        j += 1
        if j > 500 or k > 30:
            j_list.append(j)
            k_list.append(k)
            keyOld, scoreOld, j, k = keyMax, scoreMax, 0, 0
        temp += tempDelta
        iters += 1

        if iters % 100 == 0:
            pm = progressMarker(scoreMax)
            pm = pm + 0.01 * (m - lenk)
            pm = sign(pm) * max(0.01, abs(pm))
            tempDelta = tempDeltaBase * (max(m / lenk, 1) + pm * 100)
            if iters % max(1000, 100 * round((lenk - 6) ** 4 * distance2solution(scoreMax))) == 0:
                m += 1
        if iters % 4000 == 0:
            print(f'\t<{round(scoreMax, 2)}>,\t{round(time.time() - t1, 2)} sec,\t temp = {round(temp, 2)},\t j={j}, k={k},\t pm = {round(pm, 3)},\t m={m}, \t{len(keyOld)}')

    print(f'Obliczano przez {round(time.time() - t1, 2)} sekund,\t(iters={iters},\tm={m})')
    print(f'j_list.mean = {sum(j_list) / len(j_list)},\t j_list.max = {max(j_list)}')
    print(f'k_list.mean = {sum(k_list) / len(k_list)},\t jk_list.max = {max(k_list)}')
    return [scoreMax, keyMax, hill_cipher_decrypt(ct, keyMax)]

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

    best_score = -float('inf')
    best_result = None
    for multiplier in [-2,5]:
        print(f"Testing with acceptance multiplier: {multiplier}")
        result = SimAnnealing_self_adjusting(cipher_text, key_size, acceptance_multiplier=multiplier, time_limit=60)
        if result[0] > best_score:
            best_score = result[0]
            best_result = result
        print(f"Result with multiplier {multiplier}: {result[0]}")

    best_key = best_result[1]

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
