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
Run `latent_project_msas.py --align-ref <PFam ID of "query" seq>` .
^
1. GET (request) from InterPro API
2. convert characters to numbers; i.e. enumerate the a.a.'s
3. use PyTorch to convert *each* representative number to a one-hot *vector*
   final representation fed into model will be a NumPy array, `(# sequences) x (# a.a.'s)*(# MSA's)`, one row / sequence
	- each row is just all the one-hot vectors for the sequence slapped end-to-end one after another
	- need "space" for all possible a.a. types *per MSA*

[Numpy - Converting between Chars and Ints](https://gist.github.com/tkf/2276773)

## Encoding
Each a.a. converted to a **one-hot** vector, *size = number of amino acid "types"*.
***Gaps encoded as 0'th position in one-hot vector***.
=> To the model, gaps are just another dimension of the same vector including all the amino acid