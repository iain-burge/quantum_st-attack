Supplementary Material to: A Quantum Algorithm for Assessing Node Importance in the st-Connectivity Attack
===

<a href="https://github.com/iain-burge/iain-burge">Iain Burge, Institut Polytechnique de Paris, France.</a>

<a href="https://carleton.ca/scs/people/michel-barbeau/">Michel Barbeau, Carleton University, School of Computer Science, Canada.</a>

<a href="http://j.mp/jgalfaro">Joaquin Garcia-Alfaro, Institut Polytechnique de Paris, France.</a>

## Abstract

Problems in distributed security often map naturally to graphs. The
centrality of nodes assesses the importance of nodes in a graph. It is
used in various applications. Cooperative game theory has been used to
create nuanced and flexible notions of node centrality. However, the
approach is often computationally complex to implement classically.
This work describes a quantum approach to approximating the importance
of nodes that maintain a target connection. Additionally, we detail a
method for quickly identifying high-importance nodes. The
approximation method relies on quantum subroutines for
st-connectivity, approximating Shapley values, and finding the maximum
of a list. Finding important nodes relies on a quantum algorithm for
finding the maximum. We consider st-connectivity attack scenarios in
which a malicious actor disrupts a subset of nodes to perturb the
system functionality. Our methods identify the nodes that are most
important with the aim of minimizing the impact of the attack. The
node centrality metric identifies where more redundancy is required
and can be used to enhance network resiliency. Finally, we explore the
potential complexity benefits of our quantum approach in contrast to
classical random sampling.

*Keywords:* Quantum Computing, Quantum Algorithm, Distributed System,
Graph Analytics, st-Connectivity, Game theoretic node centrality.

*Version:* January 31, 2025

### Updated code

All the code related to this work is available in <a href="https://github.com/iain-burge/quantum_st-attack/tree/main/code">this repository</a>.

### Experimental results

We conducted some python simulations depicted in <a href="https://github.com/iain-burge/quantum_st-attack/blob/main/paper/preprint-arxiv.pdf">this preprint</a>.

Consider the following network:

![](img/network.png?raw=true)



## References

If using this code for research purposes, please cite:

Iain Burge, Michel Barbeau and Joaquin Garcia-Alfaro. A Quantum Algorithm for Assessing Node Importance in the st-Connectivity Attack, 2025.

```
@misc{burge-barbeau-alfaro2025st-attack,
  title={A Quantum Algorithm for Assessing Node Importance in the st-Connectivity Attack},
  author={Burge, Iain and Barbeau, Michel and Garcia-Alfaro, Joaquin},
  pages={1--14},
  year={2025},
  month={January},
  eprint={},
  archivePrefix={arXiv},
  primaryClass={quant-ph},
  url={},
}
```



