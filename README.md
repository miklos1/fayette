[![DOI](https://zenodo.org/badge/100367823.svg)](https://zenodo.org/badge/latestdoi/100367823)

#### Experimentation framework for the manuscript:

> Miklós Homolya, Robert C. Kirby, and David A. Ham. "Exposing and
> exploiting structure: optimal code generation for high-order finite
> element methods." (2017).

#### Usage

To collect measurement data, run `measure-assembly.py` and
`measure-action.py` respectively.  For the baseline, apply the correct
patch to TSFC (`fiat-mode` branch) and call the data collection
scripts with `--mode coffee`.

These scripts append raw measurement data to:
 - `assembly.csv`
 - `matvec.csv`
 - `matfree.csv`

We have uploaded the collected datasets that were used to make the
plots in the paper in this repository.  Run `generate.py` to process
raw data and produce the files which are directly pulled in by the TeX
file to render the paper.  `generate.py` creates all these files in
`data/`; this directory should be copied to the LaTeX source
directory.
