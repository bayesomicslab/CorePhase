# CorePHASE: Haplotype Phasing via Graph Homomorphisms

CorePHASE is a computational tool for phasing haplotypes from VCF (Variant Call Format) files using graph homomorphisms and expectation-maximization (EM).

## Installation

### Prerequisites

- Linux, or Windows 10/11 with WSL2 and an Ubuntu distribution
- Conda or Miniforge installed in that Linux environment
- Python 3.10 (the version used by the setup below)

`graph-tool` is required and is not available from conda-forge for native Windows (`win-64`). On Windows, run the commands below in your WSL2 Ubuntu terminal, not Windows PowerShell or a native Windows conda prompt.

### Setting up the conda environment

Install the compiled graph and numerical dependencies together from conda-forge. `pysam` is installed from PyPI by `requirements.txt`:

```bash
# Create the environment (run once)
conda create -n corephase -c conda-forge python=3.10 pip graph-tool numpy numba scipy line_profiler

# Activate the environment
conda activate corephase

# Install the remaining Python dependency
python -m pip install -r requirements.txt

# Verify the required modules before running
python -c "import graph_tool, numpy, numba, pysam, scipy, line_profiler"
```

For other installation methods and platform details, see the [graph-tool installation instructions](https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions).

## Usage

### Basic Usage

```bash
# Run from the repository root
python corePHASE.py --filename input.vcf --output_prefix output
```

### Input Format

The input file must be in **VCF (Variant Call Format)** or **BCF (Binary VCF)** format with:
- Called, diploid, biallelic genotypes (alleles `0` and `1`)
- Multiple samples (columns)
- Multiple variants (rows)

The VCF may be uncompressed or bgzip-compressed. Input files do not need an index. Missing, haploid, polyploid, or non-biallelic calls are rejected with an error.

Example VCF structure (tab-separated):
```
##fileformat=VCFv4.2
#CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO    FORMAT  sample1 sample2 ...
chr1    1000    .       A       G       60      PASS    .       GT      0/1     1/1
chr1    1001    .       T       C       60      PASS    .       GT      0/1     0/0
```

### Output Format

The output file is in **VCF format** with:
- Phased genotypes (GT field with `|` separator)
- Same header and samples as the input file
- Resolved haplotype pairs for each sample at each variant

Example output:
```
##fileformat=VCFv4.2
#CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO    FORMAT  sample1 sample2 ...
chr1    1000    .       A       G       60      PASS    .       GT      0|1     1|1
chr1    1001    .       T       C       60      PASS    .       GT      0|1     0|0
```

## Command-Line Arguments

```
--filename FILE                         Input VCF/BCF file (default: input.vcf)
--output_prefix PREFIX                  Output file prefix (default: "output")
--input_filetype {vcf,bcf}             Input file type (default: "vcf")
--threads N                             Number of threads for parallel processing (default: 1)
--draw                                  Draw core components as SVG files (default: False)
--em_init {proportional,random}         EM frequency initialization method:
                                        - proportional: frequencies proportional to adjacent edges (default)
                                        - random: uniform frequencies
                                        - proprandom: sampled proportionally
--em_component_size_threshold N         Trigger heuristic if component has more than N vertices (default: 10000)
--em_min_genotype_explained N           Minimum degree for haplotype pairs (default: 1)
--num_it N                              Number of EM iterations to run (default: 1)
--skipprop RATIO                        Skip components larger than RATIO × num_genotypes (default: 1)
--greedy MIN_EXPL                       Use greedy heuristic with minimum explanations threshold
--greedy_genmult MIN_EXPL              Use greedy heuristic weighted by genotype multiplicities
```

## Examples

### Standard phasing
```bash
python corePHASE.py --filename input.vcf --output_prefix sample_phased --em_init proportional
```

### Phasing with heuristics
```bash
python corePHASE.py --filename input.vcf --output_prefix sample_phased --greedy 2
```

### Parallel processing
```bash
python corePHASE.py --filename input.vcf --output_prefix sample_phased --threads 4
```

### Multiple EM iterations
```bash
python corePHASE.py --filename input.vcf --output_prefix sample_phased --num_it 3
```

## Algorithm Details

See the citation.

### Greedy Heuristics
- `--greedy`: Selects haplotype pairs maximizing the sum of degrees (number of explained genotypes)
- `--greedy_genmult`: Weights the degree sum by genotype multiplicity

## Performance Notes

- For components larger than `--em_component_size_threshold`, the optional degree-based heuristic can be enabled by setting `--em_min_genotype_explained` above 1
- Parallel processing (`--threads > 1`) significantly speeds up core processing

## Output Files

- `output.vcf`: Phased genotypes (the default)
- `core*.svg`: Visual representations of graph components (if `--draw` is enabled)

## Citation

If you use CorePHASE, please cite (TBD)
