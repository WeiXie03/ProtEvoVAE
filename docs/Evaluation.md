# "Accuracy" of Similarity
Want similar $\equiv$
- evolutionarily -- $\alpha$ "closer" in phylogenetic tree
	- e.g. two varieties of _C. difficile_ _very close_, _Acinetobacter baumanii, johnsonii_ less so, tree and pig even less
	__=> _YAES_, [[#Phylogeny Visualization]]___
- 

## Phylogeny Visualization
When plot each sequence in latent space, colour by empirical(?) phylogenetic classification

#### Implementing
Packages:
Python
- Python trees toolkit, [ETE Toolkit](http://etetoolkit.org/)
- [retrieve taxonomy information from NCBI taxon ID via UniProtKB SPARQL Python API](https://github.com/sib-swiss/sparql-training/blob/master/uniprot/04_taxonomy.ipynb)
- new package by Shen Wei, [TaxonKit](https://bioinf.shenwei.me/taxonkit/)
R
- [taxize](https://docs.ropensci.org/taxize/), [taxizedb](https://docs.ropensci.org/taxizedb/)
 
#### Next step
Show circular phylogenetic tree beside latent plot, colour on a gradient radially outwards. When select a sequence (point on the latent plot), highlight its "source" organism on the tree.

# Areas to Improve
- Currently, naive sequence-based approach learns star-shaped phylogeny for *all* families of proteins
	^-- compare AlphaFold predictions, in addition to sequences, when "determining" protein similarity