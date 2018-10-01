# The-Simpsons_Classifier
Progetto del corso "Apprendimento Automatico" a.a. 2017/2018 

## Scopo del progetto
Il progetto vuole analizzare la precisione di alcuni algoritmi di classificazione per l'analisi di testo.
Il dataset utilizzato rappresenta i dati della serie TV "I Simpson", in particolare si andrà ad analizzare le battute di
copione delle puntate di questa serie e l'algoritmo, data una frase, dovrà individuare quale personaggio della serie TV
possa averla detta.

## Sorgente dei dati
I dati utilizzati sono stati scaricati da [Kaggle](https://bit.ly/2M50wtO).
Il file utilizzato è *simpsons_script_lines.csv*.


**N.B.**


L'importazione dei dati presenti nel file *simpsons_script_lines.csv* presentava diversi warning dovuti ad errori nel file
come ad esempio la mancanza di alcune virgolette. Il programma funziona ugualmente, stampando solo errori, ma per
evitare queste segnalazioni è consigliato utilizzare il file *Data/simpsons_script_lines.csv* presente nella cartella del
progetto che è stato corretto manualmente.

## Librerie necessarie
- [Sklearn](http://scikit-learn.org/stable/index.html)
- [Scipy](https://www.scipy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Pandas_ml](https://pypi.org/project/pandas_ml/)
- [Numpy](http://www.numpy.org/)
- [NLTK](https://www.nltk.org/)
