CS613 - HW3 - Alex Lapinski

In order to run the code contained within this homework submission first make sure the following are available.
 * Python 2.7 (but not python 3)
 * Windowing System (Matplotlib doesn't always work well with python virtual environments)
 * make

Then run the following commands using Make:

To setup & install dependencies run one of the following:
    * make
    * pip install -r requirements.txt

To run the Naive Bayes Classifier example run one of the following (from the hw3 directory):
    * make part2
    * python src/hw3.py --naive-bayes --data ./spambase.data

To run the Multi-Class SVM example run one of the following (from the hw3 directory):
    * make part3
    * python src/hw3.py --svm --data ./CTG.csv

I've included a help feature of the hw3.py module, just run "python src/hw3.py -h".