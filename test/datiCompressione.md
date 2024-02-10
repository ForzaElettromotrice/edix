# Test performance

Di seguito verranno elencati dei testAccuracy effettuati sulle funx. I testAccuracy vengono eseguiti dalla funzione
testFunc. Per **ogni funx**, la funx verrà eseguita **100** volte per ogni numero di thread **x ∈ [2, 20]**

## Dati compressione (da finire, pc Federico)

### immagine.ppm = 691200 Byte**

*Legenda*:

* **CImg** = Immagine compressa
* **i** = Grandezza del blocco minimo per comprimere

Applicazione algoritmo lz...

| CImg (byte) | i  |
|-------------|----|
| 2035987     | 1  |
| 1414769     | 2  |
| 913361      | 3  |
| 778466      | 4  |
| 729601      | 5  |
| 666231      | 6  |
| 651803      | 7  |
| 646639      | 8  |
| 636623      | 9  |
| 634925      | 10 |
| 634252      | 11 |
| 633096      | 12 |
| 632977      | 13 |
| 632984      | 14 |
| 633195      | 15 |
| 633328      | 16 |
| 633446      | 17 |
| 633784      | 18 |
| 633869      | 19 |
| 633957      | 20 |
| 634251      | 21 |
| 634329      | 22 |
| 634427      | 23 |
| 634726      | 24 |
| 634845      | 25 |
| 634870      | 26 |
| 634961      | 27 |
| 634976      | 28 |
| 635007      | 29 |
| 635051      | 30 |
| 635127      | 31 |
| 635237      | 32 |

<!-- 
* Compressed img = 2035987 Byte   i = 1
* Compressed img = 1414769 Byte   i = 2
* Compressed img = 913361 Byte    i = 3
* Compressed img = 778466 Byte    i = 4
* Compressed img = 729601 Byte    i = 5
* Compressed img = 666231 Byte    i = 6
* Compressed img = 651803 Byte    i = 7
* Compressed img = 646639 Byte    i = 8
* Compressed img = 636623 Byte    i = 9
* Compressed img = 634925 Byte    i = 10
* Compressed img = 634252 Byte    i = 11
* Compressed img = 633096 Byte    i = 12
* Compressed img = 632977 Byte    i = 13
* Compressed img = 632984 Byte    i = 14
* Compressed img = 633195 Byte    i = 15
* Compressed img = 633328 Byte    i = 16
* Compressed img = 633446 Byte    i = 17
* Compressed img = 633784 Byte    i = 18
* Compressed img = 633869 Byte    i = 19
* Compressed img = 633957 Byte    i = 20
* Compressed img = 634251 Byte    i = 21
* Compressed img = 634329 Byte    i = 22
* Compressed img = 634427 Byte    i = 23
* Compressed img = 634726 Byte    i = 24
* Compressed img = 634845 Byte    i = 25
* Compressed img = 634870 Byte    i = 26
* Compressed img = 634961 Byte    i = 27
* Compressed img = 634976 Byte    i = 28
* Compressed img = 635007 Byte    i = 29
* Compressed img = 635051 Byte    i = 30
* Compressed img = 635127 Byte    i = 31
* Compressed img = 635237 Byte    i = 32
-->

### tmp.ppm = 6858432 Byte

Applicazione algoritmo lz...

| CImg (byte) | i  |
|-------------|----|
| 11646086    | 1  |
| 8730998     | 2  |
| 6431532     | 3  |
| 6041256     | 4  |
| 5905238     | 5  |
| 5400568     | 6  |
| 5338990     | 7  |
| 5315918     | 8  |
| 5172392     | 9  |
| 5158480     | 10 |
| 5154236     | 11 |
| 5136020     | 12 |
| 5133714     | 13 |
| 5134015     | 14 |
| 5166740     | 15 |

<!--
* Compressed img = 11646086 Byte  i = 1
* Compressed img = 8730998 Byte   i = 2
* Compressed img = 6431532 Byte   i = 3
* Compressed img = 6041256 Byte   i = 4
* Compressed img = 5905238 Byte   i = 5
* Compressed img = 5400568 Byte   i = 6
* Compressed img = 5338990 Byte   i = 7
* Compressed img = 5315918 Byte   i = 8
* Compressed img = 5172392 Byte   i = 9
* Compressed img = 5158480 Byte   i = 10
* Compressed img = 5154236 Byte   i = 11
* Compressed img = 5136020 Byte   i = 12
* Compressed img = 5133714 Byte   i = 13
* Compressed img = 5134015 Byte   i = 14
* Compressed img = 5166740 Byte   i = 15
-->

## Test performance funx

### Composition (da pc Andrea)

*Legenda*:

* **threads** = il numero di threads
* **time** = tempo di esecuzione in millisecondi
* **S** = Speedup
* **E** = Efficiency

*Esecuzione seriale*: **33ms**

*Esecuzioni parallele*:

| threads | time(ms) | S    | E       |
|---------|----------|------|---------|
| 2       | 11.57    | 2.85 | 142.61% |
| 3       | 8.48     | 3.89 | 129.72% |
| 4       | 8.78     | 3.76 | 93.96%  |
| 5       | 11.30    | 2.92 | 58.41%  |
| 6       | 9.44     | 3.50 | 58.26%  |
| 7       | 10.80    | 3.06 | 43.65%  |
| 8       | 8.24     | 4.00 | 50.06%  |
| 9       | 8.50     | 3.88 | 43.14%  |
| 10      | 8.48     | 3.89 | 38.92%  |
| 11      | 9.34     | 3.53 | 32.12%  |
| 12      | 11.01    | 3.00 | 24.98%  |
| 13      | 8.72     | 3.78 | 29.11%  |
| 14      | 8.32     | 3.97 | 28.33%  |
| 15      | 8.80     | 3.75 | 25.00%  |
| 16      | 8.64     | 3.82 | 23.87%  |
| 17      | 8.43     | 3.91 | 23.03%  |
| 18      | 8.04     | 4.10 | 22.80%  |
| 19      | 8.35     | 3.95 | 20.80%  |
| 20      | 9.26     | 3.56 | 17.82%  |
