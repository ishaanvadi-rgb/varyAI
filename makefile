dev:
	poetry run uvicorn backend.main:app --reload

reset-profile:
	rm -f varyai.db

install:
	poetry install

.PHONY: dev reset-profile install