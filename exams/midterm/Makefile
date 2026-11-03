# Build the midterm exam (student paper) and its answer key.
#   make        -> midterm.pdf and midterm_key.pdf
#   make clean  -> remove LaTeX auxiliary files (keeps the PDFs)
# The exam class needs two passes to print "Page x of y" and the total points.

all: midterm.pdf midterm_key.pdf

midterm.pdf: midterm.tex
	pdflatex -interaction=nonstopmode -halt-on-error midterm.tex
	pdflatex -interaction=nonstopmode -halt-on-error midterm.tex

midterm_key.pdf: midterm_key.tex midterm.tex
	pdflatex -interaction=nonstopmode -halt-on-error midterm_key.tex
	pdflatex -interaction=nonstopmode -halt-on-error midterm_key.tex

clean:
	rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk

.PHONY: all clean
