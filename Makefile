install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

create-result-directories:
	mkdir plots data