# Filename without extension
filename = main

pdf:
	pdflatex --shell-escape ${filename}
	bibtex ${filename}||true 
	pdflatex --shell-escape ${filename}
	pdflatex --shell-escape ${filename}
	open ${filename}.pdf

clean:
	rm -f ${filename}.{log,aux,out,bbl,blg,pyg,toc}
	rm -f sections/*.aux

cleanup:
	rm -f ${filename}.{pdf,log,aux,out,bbl,blg,pyg,toc}
	rm -f sections/*.aux

tarball:
	mkdir ../waseem_zeerak/
	cp -r ../code waseem_zeerak/
	mv ../README waseem_zeerak/
	mv main.pdf ../waseem_zeerak/report.pdf
	tar -czvf waseem_zeerak.tar.gz ../waseem_zeerak
