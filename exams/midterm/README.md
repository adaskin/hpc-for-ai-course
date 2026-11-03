# Midterm exam: YZM 513 High-Performance Programming for AI (Weeks 1-6)

| File | What it is |
|------|------------|
| `midterm.tex` | Exam source (LaTeX `exam` class) |
| `midterm_key.tex` | Answer-key wrapper: same source with `\printanswers` on |
| `midterm.pdf` | Student paper (3 pages of questions + 1 page formula sheet) |
| `midterm_key.pdf` | Answer key, with a short "Why" under every question |
| `Makefile` | `make` builds both PDFs, `make clean` removes aux files |

**Format:** 25 questions, 4 points each, 100 points. 13 multiple choice (Q1-Q13), 12 fill in the blank (Q14-Q25).
Time slot: 60 minutes (change `\examminutes` in `midterm.tex`). Closed book; the formula sheet on the last page is provided.

**Coverage:** Week 1: 4 questions, Week 2: 4, Week 3: 4, Week 4: 4, Week 5: 4, Week 6: 5.

**Build:** needs `pdflatex` and the `exam` class (TeX Live: `texlive-latex-extra`). Run `make` in this folder.

> The answer key and the exam source (which contains the answers) are in this folder.
> Do not publish this folder to students before the exam.
