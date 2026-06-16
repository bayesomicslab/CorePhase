# CorePHASE: Haplotype Phasing via Graph Homomorphisms

CorePHASE is a computational tool for phasing haplotypes from VCF (Variant Call Format) files using graph homomorphisms and expectation-maximization (EM).

## Installation

### Prerequisites

- Python 3.8+
- graph-tool (requires special installation - see below)

### Setting up the conda environment

Graph-tool has specific installation requirements. It's recommended to install it through conda:

```bash
# Create conda environment with graph-tool
conda create -n corephase -c conda-forge graph-tool

# Activate the environment
conda activate corephase

# Install additional dependencies from requirements.txt
pip install -r requirements.txt
```

Alternatively, for more details on graph-tool installation, see: https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions

## Usage

### Basic Usage

```bash
python corePHASE.py --filename input.vcf --output_prefix output
```

### Input Format

The input file must be in **VCF (Variant Call Format)** or **BCF (Binary VCF)** format with:
- Unphased genotypes (GT field with `/` separator)
- Multiple samples (columns)
- Multiple variants (rows)

Example VCF structure:
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
--filename FILE                         Input VCF/BCF file (required)
--output_prefix PREFIX                  Output file prefix (default: "output")
--input_filetype {raw,vcf}             Input file type (default: "vcf")
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
python corePHASE.py --filename sample.vcf --output_prefix sample_phased --em_init proportional
```

### Phasing with heuristics
```bash
python corePHASE.py --filename sample.vcf --output_prefix sample_phased --greedy 2
```

### Parallel processing
```bash
python corePHASE.py --filename sample.vcf --output_prefix sample_phased --threads 4
```

### Multiple EM iterations
```bash
python corePHASE.py --filename sample.vcf --output_prefix sample_phased --num_it 3
```

## Algorithm Details

See the citation.

### Greedy Heuristics
- `--greedy`: Selects haplotype pairs maximizing the sum of degrees (number of explained genotypes)
- `--greedy_genmult`: Weights the degree sum by genotype multiplicity

## Performance Notes

- For large components (>10,000 vertices), a heuristic is automatically applied
- Parallel processing (`--threads > 1`) significantly speeds up core processing

## Output Files

- `output.vcf` or `output.bcf`: Phased genotypes
- `core*.svg`: Visual representations of graph components (if `--draw` is enabled)

## Citation

If you use CorePHASE, please cite (TBD)
