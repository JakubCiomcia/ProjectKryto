import random
import string
import time
from typing import List, Union
from ngram_score import NgramScore
from double_playfair import encrypt, decrypt

alphabet = "ABCDEFGHIKLMNOPQRSTUVWXYZ"  # 'J' is typically merged with 'I'

ns = NgramScore('english_bigrams.txt')


def generate_random_key():
    alphabet = list(string.ascii_uppercase.replace('J', ''))
    key_part = random.sample(alphabet, 5)
    for letter in key_part:
        alphabet.remove(letter)
    key = key_part + alphabet
    print(key)
    return ''.join(key_part), ''.join(key)


def change_key(key: str) -> str:
    key_list = list(key)
    idx1, idx2 = random.sample(range(len(key_list)), 2)
    key_list[idx1], key_list[idx2] = key_list[idx2], key_list[idx1]
    return ''.join(key_list)


def HillClimbing(ciphertext: str, timelimit: int = 30) -> List[Union[int, str]]:
    keyOld1, fullKey1 = generate_random_key()
    keyOld2, fullKey2 = generate_random_key()
    decrypted_text = decrypt(ciphertext, fullKey1, fullKey2)

    if not isinstance(decrypted_text, str):
        raise ValueError("Decryption failed to return a string.")

    scoreOld = ns.score(decrypted_text)
    t1 = time.time()
    print('wspinanie się')

    while time.time() - t1 < timelimit:
        keyNew1 = change_key(keyOld1)
        keyNew2 = change_key(keyOld2)
        fullKey1 = keyNew1 + fullKey1[5:]
        fullKey2 = keyNew2 + fullKey2[5:]
        decrypted_text = decrypt(ciphertext, fullKey1, fullKey2)

        if not isinstance(decrypted_text, str):
            continue

        scoreNew = ns.score(decrypted_text)
        if scoreNew > scoreOld:
            keyOld1 = keyNew1
            keyOld2 = keyNew2
            scoreOld = scoreNew
            print(f'scoreOld = {scoreOld}')

    return [scoreOld, keyOld1, keyOld2, decrypt(ciphertext, fullKey1, fullKey2)]
