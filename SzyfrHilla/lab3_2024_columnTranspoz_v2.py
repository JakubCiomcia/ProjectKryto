from sys import exit as sys_exit
import random
import time
from ngram_score import NgramScore
from multiprocessing import cpu_count as mp_cpu_count
from multiprocessing import Pool as mp_pool
import math

alfabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ns = NgramScore("english_bigrams.txt")

def generateKey( lenk):
    return random.sample( list(range(lenk)), lenk )
def column( matrix, i, numLongerColumns):
    lenCol = len(matrix) if i < numLongerColumns else len(matrix)-1
    return ''.join( [ matrix[j][i] for j in range( lenCol )] )
def encrypt( M, key): # key is a list of positions, like `key=[2,0,3,1]`
    lenk = len(key)
    matrixHeight= len(M)//lenk if len(M)%lenk==0 else len(M)//lenk + 1
    numLongerColumns = lenk if len(M) % lenk==0 else len(M)%lenk
    print( f'len(M)={len(M)}, lenk={lenk}, matrixHeight={matrixHeight}, numLongerColumns={numLongerColumns}')
    #print( f'lenColumns = {[lenColumns.values()]}')
    matrix = [ M[ lenk*i : lenk*(i+1)] for i in range( matrixHeight) ]
    #print()
    #for row in matrix:
    #    print( row )
    C = ''
    for columnNumber in key:
        C += column( matrix, columnNumber, numLongerColumns)
    return C
def decrypt( C, key):
    lenk = len(key)
    matrixHeight= len(C)//lenk if len(C)%lenk==0 else len(C)//lenk + 1
    numLongerColumns = lenk if len(C)%lenk==0 else len(C)%lenk
    lenColumns = { i: matrixHeight if i<numLongerColumns else matrixHeight-1 for i in range(lenk) }
    matrix = [ ['']*lenk for i in range(matrixHeight) ]
    columns, currPos = {},0
    for columnNumber in key:
        columns[ columnNumber] = C[ currPos : currPos + lenColumns[ columnNumber] ]
        currPos += lenColumns[ columnNumber]
    for columnNumber in range(lenk):
        for il,letter in enumerate( columns[ columnNumber]):
            matrix[ il][ columnNumber] = letter
    M = ''.join([ ''.join(matrix[i]) for i in range(len(matrix)) ])
    #for row in matrix:
    #    print( ''.join(row))
    return M

def swap2(key):
    r1, r2 = sorted (random.sample (list (range (len (key))), 2))
    return key[: r1] + [key[r2]] + key[r1 + 1 : r2] + [key[r1]] + key[r2 + 1 :]
    #return generateKey( lenk)   # Monte-Carlo BruteForce
# inverseKey, shiftKey, swap3, inverseFragmentKey, shiftFragmentKey
def inverseKey(key):
    return key[::-1]
def shiftKey(key):
    r = random.randint( 0, len(key)-1)
    return key[r:] + key[:r]
def swap3(key):
    r1, r2, r3 = sorted (random.sample (list (range (len (key))), 3))
    if random.random() < 0.5:
        return key[: r1] + [key[r3]] + key[r1 + 1 : r2] + [key[r1]] \
               + key[r2 + 1 : r3] + [key[r2]] + key[r3 + 1 :]
    else:
        return key[: r1] + [key[r2]] + key[r1 + 1 : r2] + [key[r3]] \
               + key[r2 + 1 : r3] + [key[r1]] + key[r3 + 1 :]
def inverseFragmKey(key):
    r1, r2 = sorted (random.sample (list (range (len (key))), 2))
    while r2-r1 < 3 or r2-r1 > 10:
        r1, r2 = sorted (random.sample (list (range (len (key))), 2))
    return key[:r1] + inverseKey( key[r1:r2] ) + key[r2:]
def shiftFragmKey(key):
    r1, r2 = sorted (random.sample (list (range (len (key))), 2))
    while r2-r1 < 3 or r2-r1 > 10:
        r1, r2 = sorted (random.sample (list (range (len (key))), 2))
    return key[:r1] + shiftKey( key[r1:r2] ) + key[r2:]
def shiftKeyValues(key):
    r = random.randint( 0, len(key)-1)
    return [ (val+r)%len(key) for val in key]
def lengthenKey(key):
    r = random.choice( list(range(len(key))))
    return key[:r] + [ len(key)] + key[r:]
def shortenKey(key):
    if len(key) > 6:
        r = key.index(len(key)-1)
        return key[:r] + key[r+1:]
    else:
        return key
def lengthenKeyMany(key):
    r = random.choices( [2,3,4,5,6], weights=[0.3,0.3,0.2,0.1,0.1] )[0]
    key_new = lengthenKey(key)
    i = 1
    while i < r:
        key_new = lengthenKey(key_new)
        i += 1
    return key_new
def shortenKeyMany(key):
    r = random.choices( [2,3,4,5,6], weights=[0.3,0.3,0.2,0.1,0.1] )[0]
    if len(key) > r+3:
        key_new = shortenKey(key)
        i = 1
        while i < r:
            key_new = shortenKey(key_new)
            i += 1
        return key_new
    else:
        return key

def changeKeyFull(key):
    r = random.random()
    r_probs = [ 0.005, 0.035, 0.03, 0.03, 0.1, 0.02, 0.02, 0.02, 0.01, 0.01, 0.72  ]
    if r < sum(r_probs[:1]):
        return inverseKey(key) #, 'inverseKey'
    elif r < sum(r_probs[:2]):
        return shiftKey(key) #, 'shiftKey'
    elif r < sum(r_probs[:3]):
        return inverseFragmKey(key) #, 'inverseFragmKey'
    elif r < sum(r_probs[:4]):
        return shiftFragmKey(key) #, 'shiftFragmKey'
    elif r <  sum(r_probs[:5]):
        return swap3(key) #, 'swap3'
    elif r <  sum(r_probs[:6]):
        return shiftKeyValues(key) #, 'shiftKeyValues'
    elif r <  sum(r_probs[:7]):
        return lengthenKey(key) #, 'lengthenKey'
    elif r <  sum(r_probs[:8]):
        return shortenKey(key) #, 'shortenKey2'
    elif r <  sum(r_probs[:9]):
        return lengthenKeyMany(key) #, 'lengthenKey'
    elif r <  sum(r_probs[:10]):
        return shortenKeyMany(key) #, 'shortenKey2'
    else:
        return swap2(key) #, 'swap2'

def changeKey(key):
    #return changeKey_SFK(key)
    return changeKeyFull(key)

def HillClimbing(ct, lenk, timelimit = 30, spam= False):
    # można ją zrobić mniej sztywną, restartując z nowego losowego klucza
    # co jakiś czas lub co jakąś ilość iteracji lub po kilku tysięcach nieudanych prób ulepszenia
    keyOld = generateKey( lenk)
    scoreOld = ns.score (decrypt (ct, keyOld))
    t1 = time.time ()
    if spam:        print (' wspinanie się')
    while time.time () - t1 < timelimit :
        keyNew = changeKey (keyOld)
        scoreNew = ns.score (decrypt (ct, keyNew))
        if scoreNew > scoreOld :
            keyOld = keyNew
            scoreOld = scoreNew
            if spam:        print ( f' scoreOld = {scoreOld}') #,\t msg')
    return[scoreOld, keyOld, decrypt (ct, keyOld)]

def HillClimbingWithRestarts(ct, lenk, timelimit = 30, spam= False):
    # można ją zrobić mniej sztywną, restartując z nowego losowego klucza
    # co jakiś czas lub co jakąś ilość iteracji lub po kilku tysięcach nieudanych prób ulepszenia
    keyOld = generateKey( lenk)
    scoreOld = ns.score (decrypt (ct, keyOld))
    t1 = time.time ()
    if spam:        print (' wspinanie się z restartem')
    wyniki = []
    while time.time () - t1 < timelimit:
        if random.random() < 0.01:
            wyniki.append( [scoreOld, keyOld, decrypt (ct, keyOld)] )
            keyOld = generateKey( lenk)
            scoreOld = ns.score (decrypt (ct, keyOld))
            print('\n')
        keyNew = changeKey (keyOld)
        scoreNew = ns.score (decrypt (ct, keyNew))
        if scoreNew > scoreOld :
            keyOld = keyNew
            scoreOld = scoreNew
            if spam:        print ( scoreOld, end='\t')
    wyniki.sort()
    wyniki.reverse()

    return wyniki[0],

def SGHC( ct, lenk, wait = 1, timelimit = 10, spam = False):
    wyniki = []
    t1 = time.time()        # from time import time as tm; t1 = tm()
    if spam:        print('SGHC')
    while  time.time() - t1 < timelimit:    # funkcja szuka klucza góra 'timelimit' sekund
        [scoreOld, keyOld, encrypted] = HillClimbing(ct, lenk, timelimit = wait)
        if spam:
            print([scoreOld, encrypt(ct, keyOld), keyOld])
        wyniki.append( [scoreOld, keyOld, encrypted] )
    wyniki.sort()
    wyniki.reverse()
    print()
    return wyniki

def SGHC_MP_min( ct, lenk, wait = 1, spam = False):
    t1 = time.time()        # from time import time as tm; t1 = tm()
    if spam:        print('SGHC_MP minimal')
    ncore = mp_cpu_count()
    with mp_pool() as pool:
        chunks = 10*(ncore-1)
        iterable = [ [ct, lenk, wait] for i in range(chunks) ]
        results = pool.starmap( HillClimbing, iterable, chunksize= 10 )
    results.sort( key = lambda x: x[0] )
    results.reverse()
    t2 = time.time()
    print( f'SGHC_MP evaluated in {round(t2-t1,2)} sec and used {chunks} threads')
    return results

def SGHC_MP_timelimit( ct, lenk, wait = 1, timelimit = 10, spam = False):
    t1 = time.time()        # from time import time as tm; t1 = tm()
    if spam:        print('SGHC_MP with timelimit')
    ncore = mp_cpu_count()
    printedValue, wyniki, uzyte_watki = -9e99, [], 0
    while  time.time() - t1 < timelimit:    # funkcja szuka klucza góra 'timelimit' sekund
        #def genParams( params, n_chunks):
        #    for i in range(n_chunks):
        #        yield params
        with mp_pool() as pool:
            n_chunks = 10*(ncore-1)
            #iterable = genParams( [ct, lenk, wait], n_chunks)
            iterable = [ [ct, lenk, wait] for i in range(n_chunks) ]
            results = pool.starmap( HillClimbing, iterable, chunksize= 10 )
            uzyte_watki += n_chunks
        wyniki = wyniki + results
        wyniki.sort()
        wyniki.reverse()
        if wyniki[0][0] > printedValue:
            if spam:        print( wyniki[0][0], end='\t')
            printedValue = wyniki[0][0]
        else:
            if spam:        print( '<', end=' ')
    t2 = time.time()
    print( f'\nSGHC_MP evaluated in {round(t2-t1,2)} sec and used {uzyte_watki} threads')
    return wyniki

def AcceptanceFunction(valueOld, valueNew, temp):
    if random.random() < math.exp( -3*(valueOld - valueNew)/temp):  # dla temp_startowej trzeba starać się, żeby prawd. było pomiędzy 0.2 a 0.7 dla umiarkowanego pogorszenia
        #można podzielić `value` przez długość kryptotekstu
        return True
    else:
        return False
def SimAnnealing_min( ct, lenk, tempDelta= -0.0003):
    t1 = time.time()
    starttemp = 100
    endtemp = 1
    temp = starttemp
    #tempDelta = -0.0003

    keyOld = generateKey( lenk)
    scoreOld = ns.score( decrypt(ct, keyOld))
    keyMax, scoreMax = keyOld, scoreOld

    temp = starttemp
    while temp >= endtemp:
        keyNew = changeKey(keyOld)
        scoreNew = ns.score( decrypt(ct, keyNew))
        if scoreNew > scoreOld:
            keyOld, scoreOld = keyNew, scoreNew
            if scoreOld > scoreMax:     # w 'scoreMax' zapamiętujemy najlepszy wynik przejścia
                keyMax, scoreMax =  keyOld, scoreOld
                print( scoreOld) #,'\t', msg )
        elif AcceptanceFunction( scoreOld, scoreNew, temp):
            keyOld, scoreOld = keyNew, scoreNew
        temp += tempDelta

    print('zatracono ', time.time()-t1, ' sekund')
    #return [keyOld, scoreOld, decrypt(ct, keyOld)]
    return [ scoreMax, keyMax, decrypt(ct, keyMax)]

def SimAnnealing_returning( ct, lenk, tempDelta = -0.0005):
    t1 = time.time()
    starttemp = 100
    endtemp = 1
    temp = starttemp
    #tempDelta = -0.0005

    keyOld = generateKey( lenk)
    scoreOld = ns.score( decrypt(ct, keyOld))
    keyMax, scoreMax = keyOld, scoreOld

    temp = starttemp
    j, j_list = 0, []
    while temp >= endtemp:
        keyNew = changeKey(keyOld)
        scoreNew = ns.score( decrypt(ct, keyNew))
        if scoreNew > scoreOld:
            keyOld, scoreOld = keyNew, scoreNew
            if scoreOld > scoreMax:     # w 'scoreMax' zapamiętujemy najlepszy wynik przejścia
                j_list.append(j)
                keyMax, scoreMax, j =  keyOld, scoreOld, 0
                print( scoreOld) #,'\t', msg )
        elif AcceptanceFunction( scoreOld, scoreNew, temp):
            if abs( scoreOld- scoreNew) > 40:
                print( f'{scoreOld} -> {scoreNew}')
            keyOld, scoreOld = keyNew, scoreNew
        j += 1
        if j > 100:
            keyOld, scoreOld, j = keyMax, scoreMax, 0
        temp += tempDelta

    print('zatracono ', time.time()-t1, ' sekund')
    print( f'j_list.mean = {sum(j_list)/len(j_list)},\t j_list.max = {max(j_list)}')
    #return [ scoreOld, keyOld, decrypt(ct, keyOld)]
    return [ scoreMax, keyMax, decrypt(ct, keyMax), j_list]

def SimAnnealing_self_adjusting( ct, lenk, tempDeltaBase = -0.001):
    t1 = time.time()
    starttemp = 100
    endtemp = 1
    temp = starttemp
    tempDelta = tempDeltaBase

    keyOld = generateKey( lenk)
    scoreOld = ns.score( decrypt(ct, keyOld))
    keyMax, scoreMax = list(keyOld), float(scoreOld)

    ctScore, iters = ns.score(ct), 0
    def distance2solution( currScore, lenText=len(ct), ctScore= ctScore):
        return ( -2.35 - currScore/lenText) / (-2.35 - ctScore/lenText)
    def progressMarker( currScore):
        pm = ( temp/starttemp) - distance2solution( currScore)
        return pm

    temp = starttemp
    j,k, j_list, k_list, restarts = 0, 0, [], [], max( 1, lenk-10)
    m = 0 #lenk // 2

    def sign(a):
        return bool(a > 0) - bool(a < 0)

    while temp >= endtemp:
        keyNew = changeKey( keyOld)
        scoreNew = ns.score( decrypt(ct, keyNew))
        if scoreNew > scoreOld:
            keyOld, scoreOld = keyNew, scoreNew
            k += 1
            if scoreOld > scoreMax: # w 'scoreMax' zapamiętujemy najlepszy wynik
                j_list.append(j); k_list.append(k)
                print( f'{scoreOld},\t{round(time.time()-t1,2)} sec,\t temp = {round(temp,2)},\t j={j}, k={k}, \t{len(keyOld)}') #, \t {msg}' )
                keyMax, scoreMax, j, k =  keyOld, scoreOld, 0, 0
        elif AcceptanceFunction( scoreOld, scoreNew, temp):
            keyOld, scoreOld = keyNew, scoreNew
            k += 1
        j += 1
        if j > 500 or k > 30:
            j_list.append(j); k_list.append(k)
            keyOld, scoreOld, j, k = keyMax, scoreMax, 0, 0
        temp += tempDelta
        iters += 1

        if iters%100 == 0:
            pm = progressMarker( scoreMax)
            pm = pm + 0.01*(m-lenk)
            pm = sign(pm) * max( 0.01, abs(pm) )
            tempDelta = tempDeltaBase * ( max(m/lenk,1) + pm*100 )
            if iters%max( 1000, 100*round( (lenk-6)**4 * distance2solution(scoreMax)))==0:
                m += 1
                #if abs(pm) == 0.01:
                #    temp += 50
        if iters%50000 == 0:
            print( f'\t<{round(scoreMax,2)}>,\t{round(time.time()-t1,2)} sec,\t temp = {round(temp,2)},\t j={j}, k={k},\t pm = {round(pm,3)},\t m={m}, \t{len(keyOld)}')

    print( f'Obliczano przez {round(time.time()-t1,2)} sekund,\t(iters={iters},\tm={m})')
    print( f'j_list.mean = {sum(j_list)/len(j_list)},\t j_list.max = {max(j_list)}')
    print( f'k_list.mean = {sum(k_list)/len(k_list)},\t jk_list.max = {max(k_list)}')    #print( f'tempDelta = {tempDelta}')
    #return [keyOld, scoreOld, decrypt(ct, keyOld)]
    return [scoreMax, keyMax, decrypt(ct, keyMax)]

def SGSA_MP_timelimit( ct, lenk, tempDelta=-0.01, timelimit = 20, spam = False):
    t1 = time.time()        # from time import time as tm; t1 = tm()
    if spam:        print('SGHC_MP with timelimit')
    ncore = mp_cpu_count()
    printedValue, wyniki, uzyte_watki = -9e99, [], 0
    while  time.time() - t1 < timelimit:    # funkcja szuka klucza góra 'timelimit' sekund
        #def genParams( params, n_chunks):
        #    for i in range(n_chunks):
        #        yield params
        with mp_pool() as pool:
            n_chunks = 3*(ncore-1)
            #iterable = genParams( [ct, lenk, wait], n_chunks)
            iterable = [ [ct, lenk, tempDelta] for i in range(n_chunks) ]
            #results = pool.starmap( SimAnnealing_returning, iterable, chunksize= 1 )
            results = pool.starmap( HillClimbing, iterable, chunksize= 1 ) # HillClimbing
            uzyte_watki += n_chunks
        wyniki = wyniki + results
        wyniki.sort()
        wyniki.reverse()
        if wyniki[0][0] > printedValue:
            if spam:        print( wyniki[0][0], end='\t')
            printedValue = wyniki[0][0]
        else:
            if spam:        print( '<', end=' ')
    t2 = time.time()
    print( f'\nSGHC_MP evaluated in {round(t2-t1,2)} sec and used {uzyte_watki} threads')
    return wyniki


if __name__ == '__main__':
    print( 'Logical processors: ', mp_cpu_count() )

    text = " Data Repository is free-to-use and open access. It enables you to deposit any research data (including raw and processed data, video, code, software, algorithms, protocols, and methods) associated with your research manuscript. Your datasets will also be searchable on Mendeley Data Search, which includes nearly 11 million indexed datasets. For more information, visit ."
    text = """Last night I had a deeply distressing dream. I was coming through a forest and I lost my way. The weather was awful. It was dark and chilly. I was shivering with cold. After three hours of wandering around, I found some old and ruinous hut. I knocked at the door but nobody opened. I pressed the handle and noticed that the door was unlocked. I entered and sat in front of the fire. Although I was a stranger I started looking for some water. When I found a small door I opened it and went into the cellar. It was very huge and damp. I heard some terrible noise similar to howling or growling. I found a torch and lighted it. I saw hundreds of cages with imprisoned animals. I came to the first cage and suddenly I thought, 'Oh, God, this beast is familiar to me'. In the next cage there was some big wolf with red eyes and sharp long teeth. His mouth was covered in blood. I knew it, too. I observed some other boxes and I was convinced I had seen those creatures before. Then I heard crying. I approached the source of the sound and I saw a girl. She had neither eyes nor ears. Her skin was burnt. She looked scary. I recollected that I knew those monsters from my nightmares. 'You will die', somebody murmured. I was so frightened that I hardly managed to turn back. I noticed an opened cage. There was a notice above it: 'The factory of nightmares. The last stage'. I started running away but a monstrous spider stood on my way. It approached and jumped at me. Then I felt that somebody or something was shaking my body. I opened my eyes and heard: 'Calm down, it was just a dream'."""
    text = ''.join([ c for c in text.upper() if c in alfabet ]) #[:299]
    key0 = generateKey(15)
    lenk = len(key0)
    print( 'Preprocessed Text:\n', text, '\n', ns.score(text))
    ct = encrypt( text, key0)
    print( 'Cryptotext: \n', ct, '\n', ns.score(ct))
    tj = decrypt( ct, key0)
    print( 'Decrypted: \n', tj, '\n', ns.score(tj))

    solution = SimAnnealing_self_adjusting( ct, 10)
    print( solution[:3] )
    print( key0)
