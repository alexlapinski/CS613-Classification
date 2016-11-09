init:
	pip install -r requirements.txt

clean:
	rm -f graphs/**/*.png tex/*.aux tex/*.log tex/*.synctex.gz .cache

part2:
	python src/hw3.py --naive-bayes --data ./data/spambase.data

part3:
	python src/hw3.py --svm --data ./data/CTG.csv

all: part2 part3

.PHONY: init all part2 part3 clean
