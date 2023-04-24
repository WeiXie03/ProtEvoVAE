## Databases and Formats
paper uses **pfam** *multiple sequence alignment (MSA)*
### [pfam](https://youtu.be/xeBN8HPlkpE)
Hidden Markov Model-MSA of most of the *sequences* in *UniProt*, organized into protein "families"
### UniProt
Protein *sequences*
### InterPro
*classification* of proteins
- each **entry** is a... *family*, *domain* (functional "block"/"module" within the protein), *conserved site*, etc.
	- Members:
		- matching *proteins*--what proteins have "this" in common?
		- 

*Note*: original *pfam* entry for **cytochrome P450** integrated into *InterPro* entry (accession ID) `IPR001128`

# Their Repository
## `proc_msa.py`
1. construct `seq_dict := {(seq_id, seq)}` dictionary
2. `aa_index := ` enumerate each amino acid, '-' and '.'s `:= 0`
	- *pickle* ^--> `output/aa_index.pkl`
3. 

# The Plan
### "Workflow" for User
Run `project_msa.py --align-ref <PFam ID of "query" seq>` 
- optional option: `--save_model_path <path to save trained torch model to>`
	- if provided, trains VAE further on new data and saves updated model to specified path
^
1. GET (request) from InterPro API => Stockholm (MSA) file
2. run `proc_msa.py`: Stockholm file => enumerated a.a. matrix [pickled NumPy array]
	- enumerated matrix is the MSA but just a.a. letters => numbers, 0 ~ 20
	- output, enumerated a.a. matrix, has dimensions `(# sequences) x (alignment length in # letters)`, i.e. one row / sequence
3. run 
	- use torch to convert *each* representative number to a one-hot *vector*


[Numpy - Converting between Chars and Ints](https://gist.github.com/tkf/2276773)

## Encoding
Each a.a. converted to a **one-hot** vector, *size = number of amino acid "types"*.
***Gaps encoded as 0'th position in one-hot vector***.
=> To the model, gaps are just another dimension of the same vector including all the amino acid

## MSAs
| Family ID | Family Name               | Query Seq ID             |     | 
| --------- | ------------------------- | ------------------------ | --- |
| PF00221   | Aromatic amino acid lyase | A0A7N6B928_ANATE/116-569 |     |
|           |                           |                          |     |
