.PHONY: install extract scan search clean help

PY = .venv/bin/python

help:
	@echo "make install         — create venv and install deps (run once)"
	@echo "make scan            — embed every mp3 under data/audio/"
	@echo "make extract URL=... — embed one Suno share URL"
	@echo "make extract BATCH=urls.txt — embed all URLs/paths in a file"
	@echo "make search Q='dark drill arabic'   — text query against the index"
	@echo "make search T=<track_id>            — find tracks similar to a seed"
	@echo "make clean           — drop venv and data/index/*"

install:
	./setup.sh

scan:
	$(PY) -m scripts.extract --scan-audio-dir

extract:
ifdef URL
	$(PY) -m scripts.extract "$(URL)"
else ifdef BATCH
	$(PY) -m scripts.extract --batch "$(BATCH)"
else
	@echo "usage: make extract URL=https://suno.com/song/<id>"
	@echo "       make extract BATCH=urls.txt"
endif

search:
ifdef Q
	$(PY) -m scripts.search --text "$(Q)"
else ifdef T
	$(PY) -m scripts.search --track "$(T)"
else
	@echo "usage: make search Q='dark drill arabic'"
	@echo "       make search T=<track_id>"
endif

clean:
	rm -rf .venv data/index/*
