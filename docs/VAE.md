[PyTorch can convert arrays of indices to one-hot vectors](https://sparrow.dev/pytorch-one-hot-encoding/)

__Note__: `n_aa` to feed into `torch.functional.one_hot()` $=$ `(num of a.a. types)` ___$+1$___, because $|\set{i}_{i=0}^n| = n+1$ ($0$ included in possible a.a. types as alignment gap).

# Training
### Protein Families
I propose k-fold cross validation, dividing up _each_ multiple sequence alignment (from PFAM), for...
- Each family is potentially a _very_ different distribution, each somewhat of a rough cluster within the universe of proteins.
- Additionally, the final data of each family fed into the model are multiple sequence _alignments_, heavily dependent on the query sequence _chosen_.
^ [Is it advisable to combine two dataset?](https://datascience.stackexchange.com/q/38973)
---
In this case, should generalization in the sense of being able to accurately capture the phylogenetic relationships of a new, unseen protein family given a multiple sequence alignment of the members with a relatively "common ancestral" query sequence, be expected?

# Areas to Improve
The problem with using a VAE for this task is that the only type of phylogenetic pattern the model can learn is the vague and limited capacity of the KL divergence, or regularization term, in the ELBO. Since we are _fixing_ our prior distribution (the reference we measure divergence from) as a normal, the learned phylogenetic patterns will only include those in which descendent proteins "emanate" from a common ancestor _in a normally distributed manner_, namely star-shaped phylogenies.

### ELBO Calculation
Ding et. al. calculate $\log p(x|z)$ as
```python
torch.sum(x*log_p, -1)
```
Since `x` is a _one-hot_ vector, doesn't this mean _only_ the probability assigned to the correct a.a. type for each position is considered? Should we not add a penalty for assigning probability to the incorrect ones?