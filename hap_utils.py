import gzip
from pysam import VariantFile

def open_bcf_to_read(input_vcf_file):
	# if input_vcf_file is a gzipped file
	if input_vcf_file.endswith('.gz'):
		bcf_in = VariantFile(gzip.open(input_vcf_file, 'r'))  # auto-detect input format
	else:
		bcf_in = VariantFile(input_vcf_file)  # auto-detect input format
	return bcf_in
