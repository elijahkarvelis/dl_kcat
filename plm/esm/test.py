from transformers import AutoTokenizer, EsmModel
import torch

import numpy as np
import pickle
import sys
sys.path.append('/data/karvelis03/dl_kcat/scripts/')
from prep_data import *
import re

import umap
import matplotlib.pyplot as plt





def abbreviate_aa(abbreviation):
	# Returns the single-letter amino acid abbreviation
	# corresponding to the three-letter abbreviation 
	# passed in

	table = {'ARG':'R', 'HSE':'H', 'HSD':'H', 'HSP':'H', 'HIE':'H',
			 'HID':'H', 'HIP':'H', 'HIS':'H', 'LYS':'K',
			 'ASP':'D', 'ASQ':'D', 'GLU':'E', 'GQ2':'E', 'GQU':'E',
			 'SER':'S', 'THR':'T', 'ASN':'N', 'GLN':'Q',
			 'CYS':'C', 'GLY':'G', 'PRO':'P',
			 'ALA':'A', 'VAL':'V', 'ILE':'I', 'LEU':'L', 'MET':'M',
			 'PHE':'F', 'TYR':'Y', 'TRP':'W'}

	return table[abbreviation]


def read_fasta(filename):
	# Reads and returns amino acid sequence from FASTA file filename

	with open(filename, 'r') as f:
		seq = ''.join(f.readlines()[1::]).replace('\n','')

	return seq


def read_pdb(filename):
	# Reads and returns the amino acid sequence of Chain I from 
	# PDB file filename

	with open(filename, 'r') as f:
		data = []
		for line in f:
			if line.rsplit()[-1] == 'I':
				data.append(line)

		atoms = []
		for line in data:
			# grab amino acid and position
			atoms.append(f"{line.rsplit()[4]}|{abbreviate_aa(line.rsplit()[3])}")

	residues = list(np.unique(atoms))
	
	amino_acids, positions = [], []
	for i in residues:
		amino_acids.append(i.split('|')[-1])
		positions.append(int(i.split('|')[0]))


	amino_acids = [a for _,a in sorted(zip(positions,amino_acids))]
	positions = sorted(positions)

	# for i in range(len(amino_acids)):
	# 	print (f'{amino_acids[i]} {positions[i]}')

	return ''.join(amino_acids)


def load_mutants(meta_file, variants_summary='/data/karvelis03/dl_kcat/input/variants_summary.csv'):
	# Returns a list of the mutants in a dataset meta_file
	#             Each mutation is reported in a string
	#			  as WT residue, position, then mutant 
	#			  residue, where each residue is represented
	#			  by a single-letter abbreviation. E.g., 
	#			  'T520D' substitues aspartate for WT's 
	#			  threonine at residue 520. To insert 
	#			  multiple mutations, separate each mutation
	# 			  by a hyphen; e.g., 'Q140M-T520D'

	meta = pickle.load(open(meta_file, 'rb'))
	mutations = list(np.unique(meta.variant))

	# sort by kcat
	table = pd.read_csv(variants_summary)
	table = table.loc[table['TIS raw k'].to_numpy(dtype=str) != 'nan (nan)']
	kcats = []
	for var in mutations:
		kcat = table.loc[table['Variant']==var]['TIS raw k/k_WT'].to_numpy()[0]
		kcats.append(kcat)

	mutations = [var for _,var in sorted(zip(kcats,mutations))]
	kcats = sorted(kcats)

	return mutations


def mut_seq(wt_seq, mutation):
	# Returns mutated KARI sequence as string of single letter 
	# amino acid abbreviations. The WT sequence MUST be for 
	# KARI enzyme from Spinacia oleracea, as reported in UniProt:
	#      https://www.uniprot.org/uniprotkb/Q01292/entry
	# INPUT:
	# wt_seq -- WT sequence as a string of single letter 
	#           amino acid abbreviations
	# mutation -- str of mutations to insert into 
	#			  WT sequence. Each mutation is reported 
	#			  as WT residue, position, then mutant 
	#			  residue, where each residue is represented
	#			  by a single-letter abbreviation. E.g., 
	#			  'T520D' substitues aspartate for WT's 
	#			  threonine at residue 520. To insert 
	#			  multiple mutations, separate each mutation
	# 			  by a hyphen; e.g., 'Q140M-T520D'

	if mutation == 'WT':
		return wt_seq

	mut_seq = np.array(list(wt_seq))
	for sub in mutation.split('-'):
		wt_aa, pos, mut_aa = re.split(r'(\d+)', sub)
		wt_aa = abbreviate_aa(wt_aa.upper())
		mut_aa = abbreviate_aa(mut_aa.upper())
		pos = int(pos)

		if wt_aa != mut_seq[pos-1]:
			raise ValueError("WT sequence doesn't match sequence that reported in the mutation -- double check mutation") 
	
		mut_seq[pos-1] = mut_aa

	mut_seq = ''.join(mut_seq)

	return mut_seq





if __name__ == '__main__':

	# Read in WT sequence
	wt_fasta = 'kari_wt.fasta'
	wt_seq = read_fasta(wt_fasta)


	# Load ESM-2 using Huggingface, and use it to embed WT sequence
	tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
	model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
	model.eval()


	# Create a list of variant sequences in variants dictionary
	# variants = {mutant1:sequence, mutant2:sequence, ... WT:sequence}
	meta_file = '/data/karvelis03/dl_kcat/data/total/tptrue_gsfalse_o0dot8_s1_2_3_4_5_r1_2_t-110_0_sub1000_numNone.550000-111-70metadata'
	variants = {}
	for mutation in load_mutants(meta_file):
		variants[mutation] = mut_seq(wt_seq, mutation)


	# For each variant, create an embedding
	embeddings= {}
	for var in variants:
		inputs = tokenizer(variants[var], return_tensors="pt")
		outputs = model(**inputs)
		last_hidden_states = outputs.last_hidden_state # note: equivalent to outputs[0]
		# pooled_output = outputs.pooler_output.detach().numpy() # not exactly sure when, what, or why to use pooled_output, but the EsmForSequenceClassification does NOT

		# The EsmForSequenceClassification class from HuggingFace uses only the first 
		# token's embedding for classification predictions, stating that this first 
		# token is the <s> token (equiv. to [CLS]) meant to summarize the full sequence
		embeddings[var] = last_hidden_states[:, 0, :].reshape(1,-1).detach().numpy()


	# Visualize embeddings in 2D using UMAP
	umap_fxn = umap.UMAP(random_state=333, n_neighbors=5, min_dist=0.1)

	data, labels = [], []
	for var in embeddings:
		data.append(embeddings[var])
		labels.append(var)
	data = np.vstack(data)

	print (labels)

	embedding = umap_fxn.fit_transform(data)

	for i, var in enumerate(labels):
		print (f"{var}: {embedding[i,:]}")

	cmap = plt.cm.get_cmap('bwr')
	colors = [cmap(i) for i in np.linspace(0, 1, data.shape[0])]

	plt.scatter(embedding[:, 0], embedding[:, 1], c=colors)
	plt.scatter(embedding[0, 0], embedding[0, 1], c=colors[0], label=labels[0])
	plt.scatter(embedding[-1, 0], embedding[-1, 1], c=colors[-1], label=labels[-1])
	# plt.gca().set_aspect('equal', 'datalim')
	plt.title('UMAP projection', fontsize=24)
	plt.legend()
	plt.show()


	""" Check that PDB sequence matches UniProt FASTA """
	# wt_pdb = '/data/karvelis02/osprey_toy/structures/minimized_v5.pdb'
	# pdb_seq = read_pdb(wt_pdb)
	# print (wt_seq, '\n', pdb_seq)

	# # The PDB is missing some of the N-terminus; we must confirm
	# # that the pdb_seq is a subsequence of the wt_seq
	# print (f'pdb_seq in wt_seq: {pdb_seq in wt_seq}')




