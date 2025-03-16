# nlp-roberta-project-2024l
Projekt NLP semestr 2024L - temat ROBERTA.

https://staff.elka.pw.edu.pl/~pandrusz/nlp_tematy.html

## Autorzy
- Konrad Dumin
- Dominika Ziółkiewicz


## Wymagania
Aby móc uruchomić kod należy mieć zainstalowaną werjsę python'a 3.9 a następnie w środowisku wirtualnym wykonać komendę:
```bash
pip install -r requirements.txt
```
W razie potencjalnych problemów zaleza się instalację *python3.9-dev* oraz ponowną instalację torch==1.13.1 używającego cuda11.7 (można znależć na stronie poprzednich wersji pytorch'a).

## Datasety
Wszystkie potrzebne datasety są w katalogu `data/datasets_old` folder. Zostały pobrane z [site](http://ixa2.si.ehu.eus/stswiki/index.php/Main_Page#Interpretable_STS). Jeśli zachodzi potrzeba można je pobrać i przeprocesować za pomocą:
```bash
./download_semeval_data.sh
python tools/create_csv.py
```
# Trenowanie 
Aby zacząć trening modelu z domyślnymi parametrami należy uruchomić:
```bash
python tools/train_net.py
```
Nastomiast można ustawić dowolne parametry razem z komenda wyżej, np. można ustawić parametr `max_epochs` na 10 i `learning rate` na 0.0002 wystarczy wpisać:
```bash
python tools/train_net.py SOLVER.MAX_EPOCHS 10 SOLVER.BASE_LR 0.0002
```
Wszystkie potrzebne parametry znajdują się w pliku `net_default_cfg/default_config.py`

Parametry mogą być też zaciągnięte z pliku przy użyciu `--config_file`:
```bash
python tools/train_net.py --config_file conf/roberta_config.yml
```

# Testowanie
Aby przetestować dany model należy podać do niego ścieżkę ustawiając parametr TEST.WEIGHT:
```bash
python .\tools\test_net.py TEST.WEIGHT  "output/29052021142858_model.pt"
```

**NOTE**: Istnieją 4 testowe datasety:
- images
- headline
- answers-students
- wszystkie powyższe razem (*domyślnie*)


Obliczane są metryki F1:
- F1 score dla values
- F1 score dla explanations

# Testowanie modelu przy użyciu skryptów z SemEval 
Są 2 pliki (skrypty w perlu) które służą również do testowania modelu. Znajdują się w folderze `tests`:
- evalF1_no_penalty.pl
- evalF1_penalty.pl

Aby ich użyć dane musza być w odpowiednim formacie. W tym celu należy stworzyć plik z końcówką .wa używając:
```bash
python .\tools\create_wa.py TEST.WEIGHT  "output/29052021142858_model.pt" DATASETS.TEST_WA "data/datasets/STSint.testinput.answers-students.wa"
```
Zostanie stworzony plik z predykowanymi wartościami w katalogu 'output'. Następnie można wykonać odpowiedni skrypt, np:
```bash
perl .\evalF1_penalty.pl data/datasets/STSint.testinput.answers-students.wa output/STSint.testinput.answers-student_predicted.wa --debug=0
```




