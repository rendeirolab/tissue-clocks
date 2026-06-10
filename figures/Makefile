build: pdf

clean:
	-rm pdf/*
	-rm png/*
	-rm svg/Figure*.trimmed.svg
	-rm AllFigures*.pdf
	-rm MainFigures*.pdf
	-rm SupplementaryFigures*.pdf

pdf:
	bash process.sh

.PHONY: clean pdf build
