# EdiX

Editor di immagini scritto interamente in C++

[![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![OpenMP](https://img.shields.io/badge/OpenMP-00599C?style=for-the-badge&logo=openmp&logoColor=white)](https://www.openmp.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io/)

Questa repository contiene il codice di un progetto universitario per il corso di [Ingegneria del software](https://corsidilaurea.uniroma1.it/it/view-course-details/2023/29923/20190322090929/1c0d2a0e-d989-463c-a09a-00b823557edd/f5e77c3f-84d5-4123-8b84-8a0a5c597463/15ecc655-f8c0-4c3f-afcf-da46946dcf5f/95212068-f314-40bc-b767-b909de17d286?guid_cv=f5e77c3f-84d5-4123-8b84-8a0a5c597463&current_erogata=1c0d2a0e-d989-463c-a09a-00b823557edd) e [Progettazione di sistemi Multicore](https://corsidilaurea.uniroma1.it/it/view-course-details/2023/29923/20190322090929/1c0d2a0e-d989-463c-a09a-00b823557edd/f5e77c3f-84d5-4123-8b84-8a0a5c597463/15ecc655-f8c0-4c3f-afcf-da46946dcf5f/359f0288-368b-440a-adbf-a308d381e762/f8cef9fd-6d98-4ada-b4bf-6b9b94ca6cc3)

EdiX è un editor di immagini dove l'utente può creare dei progetti. All'interno di essi, potrà aggiungere immagini ed effettuare operazioni su di esse. Le operazioni sono denominate `funX` e sono le seguenti:

* Blur
* Scala di grigi
* Upscaling/Downscaling
* Sovrapposizione
* Composizione
* Color filter

## Struttura del progetto

* `bin/` Contiene i file binari
* `obj/` Contiene i file oggetto
* `src/` Contiene il codice sorgente
  * `src/dbutils` Contiene il codice per la creazione e gestione del database PostgreSQL e Redis
  * `src/env` Contiene il codice per la gestione degli ambienti presenti nel progetto, ovvero: HOMEPAGE, PROJECT e SETTINGS
  * `src/functions` Contiene il codice delle funX
* `test/` Contiene il codice per testare le funX

Per maggiori informazioni, [qui](https://github.com/ForzaElettromotrice/edix/files/14466210/EdiX.pdf) trovate la relazione del progetto

## Prerequisiti

Prima di passare alla compilazione del progetto bisogna avere installati i seguenti programmi

* CMake
* CUDA
* PostgreSQL
* Redis


## Come compilare ed eseguire

Una volta scaricato il progetto, entrare nella directory `edix`. Al suo interno si deve creare una nuova directory, denominata build

```bash
mkdir -p build
```

Creata build, entrare all'interno di essa ed eseguire 

```bash
cmake ..
```

Eseguito il comando, all'interno di build, verrà generato il Makefile il quale permetterà di generare gli eseguibili. Quindi per compilare

```bash
make
```

A questo punto verranno generati due eseguibili

* `edix` sarà l'eseguibile principale
* `testx` se eseguito verrà fatto un test delle performance di ogni _funX_, con 1 fino al massimo numero di thread supportabili dalla macchina (**NOTA** per eseguire testx, bisogna entrare nella directory `test` ed eseguire `testx` dal suo interno)

Per eseguire uno dei due eseguibili, basterà quindi
```bash
./nome_eseguibile
```

