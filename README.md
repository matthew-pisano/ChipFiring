# On Picard groups and Jacobians of directed graphs

This repository serves as an archive for the code used for the research in this paper.

We focus on the study of Chip-Firing games and how different combinations of 
directed and undirected edges affect its winning strategies.  The paper itself can
be found with this [arXiv link](https://arxiv.org/abs/2302.10327).

## Abstract

The Picard group of an undirected graph is a finitely generated abelian
group, and the Jacobian is the torsion subgroup of the Picard group. These
groups can be computed by using the Smith normal form of the Laplacian matrix
of the graph or by using chip-firing games associated with the graph. One may
consider its generalization to directed graphs based on the Laplacian matrix.
We compute Picard groups and Jacobians for several classes of directed trees,
cycles, wheel, and multipartite graphs.

## Submissions and Conferences

This paper was accepted into the Joint Mathematics Meetings 2023 and
presented no January 6th 2023. We also plan to submit this paper to the 
Journal of Experimental Mathematics and the Journal of Combinatorial Theory.

## Documentation

The documentation for this project of course comes within the paper itself,
but also within several presentations that I had prepared over the course
of working on this project.  These can be found in the 
[Docs/](https://github.com/matthew-pisano/ChipFiring/tree/master/Docs) folder.

## Code

The code is split into several files with each having a specific role.

- The *game.py* file contains the utility classes for Graphs and Divisors
that were instrumental in representing the abstract concepts associated with
our research.

- The *algorithms.py* file contains the algorithms that we used to either
computationally prove or guide our rigorous proofs with experimental results.

- The *utils.py* file centers around the computations for the *smith normal form*.
These can be quite heavy, so I made several optimizations to the code in
the places that I could.

