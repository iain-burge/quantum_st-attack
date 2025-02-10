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

We conducted some <a href="https://www.ibm.com/quantum/qiskit">Qiskit</a> simulations over the practical example in <a href="https://arxiv.org/abs/2502.00446">this preprint</a>. The code of our simulations is available in <a href="https://github.com/iain-burge/quantum_st-attack/tree/main/code">this repository</a>.

Consider the following network:

![](img/network.png?raw=true)

By executing <a href="https://github.com/iain-burge/quantum_st-attack/blob/main/code/QuantumSTConnectivity.py">code/QuantumSTConnectivity.py</a>, we obtain the results shown below:

````{verbatim}
==========================================
target = 0 *(Node a)*
Construct Circuit -  20:48:23 2025
Evolve State      -  20:54:02 2025
Display Results   -  22:08:38 2025

    True Value:  0.21667
    Quantum:     0.22221  (Error=0.00555)
    Monte Carlo: 0.125    (Error=0.09167)

==========================================
target = 1 *(Node b)*
Construct Circuit -  22:08:38 2025
Evolve State      -  22:14:24 2025
Display Results   -  23:08:27 2025

    True Value:  0.21667
    Quantum:     0.22221   (Error=0.00555)
    Monte Carlo: 0.1875    (Error=0.02917)

==========================================
target = 2 *(Node c)*
Construct Circuit -  23:08:27 2025
Evolve State      -  23:14:12 2025
Display Results   -  01:34:54 2025

    True Value:  0.30000
    Quantum:     0.30866  (Error=0.00866)
    Monte Carlo: 0.375    (Error=0.07500)

==========================================
target = 3 *(Node d)*
Construct Circuit -  01:34:54 2025
Evolve State      -  01:40:40 2025
Display Results   -  03:17:57 2025

    True Value:  0.00000
    Quantum:     0.00000  (Error=0.00000)
    Monte Carlo: 0.0      (Error=0.00000)

==========================================
target = 4 *(Node e)*
Construct Circuit -  03:17:57 2025
Evolve State      -  03:23:50 2025
Display Results   -  05:03:27 2025

    True Value:  0.16667
    Quantum:     0.14645  (Error=0.02022)
    Monte Carlo: 0.21875  (Error=0.05208)

==========================================
target = 5 *(Node f)*
Construct Circuit -  05:03:27 2025
Evolve State      -  05:09:09 2025
Display Results   -  06:50:23 2025

    True Value:  0.05000
    Quantum:     0.03806  (Error=0.01194)
    Monte Carlo: 0.0625   (Error=0.01250)

==========================================
target = 6 *(Node g)*
Construct Circuit -  06:50:23 2025
Evolve State      -  06:56:10 2025
Display Results   -  08:16:41 2025

    True Value:  0.05000
    Quantum:     0.03806  (Error=0.01194)
    Monte Carlo: 0.09375  (Error=0.04375)
==========================================
````


where the *Quantum* output is the result of Shapley value approximation
and the *Monte Carlo* output is the result of random sampling with the same
amount of samples used by the quantum approach.

The previous execution run provides the Shapley values depicted next:

![](img/results.png?raw=true)

Note that the Shapley value of $0$ for *Node d* indicates that this node
is useless in the game. Shapley values are based on how often subsets of nodes maintain *st*-connectivity, for example, the coalition of *Node
a* and *Node b* is *st*-connected and is depicted next:

![](img/coallition.png?raw=true)

in which coalitions of nodes are represented by binary strings (cf.
<a href="https://arxiv.org/abs/2502.00446">preprint paper</a>, Section 3.4, for further details 
about this).

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
  eprint={2502.00446},
  archivePrefix={arXiv},
  primaryClass={quant-ph},
  url={https://arxiv.org/abs/2502.00446}, 
}
```



