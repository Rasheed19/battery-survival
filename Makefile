.PHONY: setup
setup: install create-result-directories

.PHONY: install
install:
	@echo "Installing dependencies..."
	pip install --upgrade pip &&\
	pip install -r requirements.txt
	pip install -e .

.PHONY: create-result-directories
create-result-directories:
	@echo "Creating working directories..."
	mkdir plots data
	@echo "Cheers! You are all set."
