from pysam import VariantFile

def open_bcf_to_read(input_vcf_file):
	return VariantFile(input_vcf_file)  # auto-detect compression and VCF/BCF format
