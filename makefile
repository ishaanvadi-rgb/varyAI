dev:
	poetry run uvicorn backend.main:app --reload

test:
	poetry run pytest tests/ -v

reset-profile:
	rm -f varyai.db
	rm -rf chroma_db/

install:
	poetry install

.PHONY: dev test reset-profile install