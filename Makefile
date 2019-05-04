## Put this Makefile in your project directory---i.e., the directory
## containing the paper you are writing. Assuming you are using the
## rest of the toolchain here, you can use it to create .html, .tex,
## and .pdf output files (complete with bibliography, if present) from
## your markdown or Rmarkdown fies. 
## -	Change the paths at the top of the file as needed.
## -    If you're just starting with a markdown file, using `make`
##      without arguments will generate html, tex, pdf, and docx 
## 	output files from all of the files with the designated Rmarkdown
##	extension. The default is `.md` but you can change this. 
## -	You can specify an output format with `make tex`, `make pdf`,  
## - 	`make html`, or `make docx`.
## -	Doing `make clean` will remove all the .tex, .html, .pdf, and .docx files 
## 	in your working directory. Make sure you do not have files in these
##	formats that you want to keep! 

## All Markdown Files in the working directory
SRC = $(wildcard *.md)

## Location of your working bibliography file
BIB = sentemb.bib

## CSL stylesheet (located in the csl folder).
CSL = apsa

## Pandoc options to use
OPTIONS = markdown+simple_tables+table_captions+yaml_metadata_block+smart

PDFS=$(SRC:.md=.pdf)
HTML=$(SRC:.md=.html)
TEX=$(SRC:.md=.tex)


all:	$(HTML)

pdf:	clean $(PDFS)
html:	clean $(HTML)
tex:	clean $(TEX)

%.html:	%.md
	pandoc -r $(OPTIONS) -w html  --template=templates/pandoc-scholar.html --css=templates/styles/pandoc-scholar.css --filter pandoc-citeproc --csl=csl/$(CSL).csl --bibliography=$(BIB) -o $@ $<

%.tex:	%.md
	pandoc -r $(OPTIONS) -w latex -s  --pdf-engine=pdflatex --template=templates/pandoc-scholar.latex --filter pandoc-citeproc --csl=csl/ajps.csl --bibliography=$(BIB) -o $@ $<

%.pdf:	%.md
	pandoc -r $(OPTIONS) -s  --pdf-engine=pdflatex --template=templates/pandoc-scholar.latex --filter pandoc-citeproc --csl=csl/$(CSL).csl --bibliography=$(BIB) -o $@ $<

clean:
	rm -f *.html *.pdf *.tex
