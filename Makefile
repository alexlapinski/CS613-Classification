init:
	pip install -r requirements.txt

clean:
	rm -f graphs/**/*.png tex/*.aux tex/*.log tex/*.synctex.gz

part2:
	python src/hw3.py --naive-bayes --data ./spambase.data

part3:
	python src/hw3.py --svm --data ./CTG.csv

test:
    nosetests

all:
	python src/hw3.py

.PHONY: init test part2 part3 clean all
