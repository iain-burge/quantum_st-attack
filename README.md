Supplementary Material to: Identifying Vulnerable Nodes and Detecting Malicious Entanglement Patterns to Handle *st*-Connectivity Attacks in Quantum Networks
===

<a href="https://github.com/iain-burge/iain-burge">Iain Burge, Institut Polytechnique de Paris, France.</a>

<a href="https://carleton.ca/scs/people/michel-barbeau/">Michel Barbeau, Carleton University, School of Computer Science, Canada.</a>

<a href="http://j.mp/jgalfaro">Joaquin Garcia-Alfaro, Institut Polytechnique de Paris, France.</a>

## Abstract


Problems in distributed system security often map naturally to graphs.
The concept of centrality assesses the importance of nodes in a graph.
It is used in various applications. Cooperative game theory has also
been used to create nuanced and flexible notions of node centrality.
However, the approach is often computationally complex to implement
classically. We describe a quantum approach to approximating the
importance of quantum nodes that maintain a target connection in a
quantum network. We detail a method for quickly identifying
high-importance nodes that can be targeted by adversaries. The
approximation method relies on quantum subroutines for
*st*-connectivity, approximating Shapley values, and finding the
maximum of a list. We consider a malicious actor targeting a subset of
nodes to perturb the system functionality. Our method identifies the
nodes that are most important in keeping nodes $s$ and $t$ connected.
Once we have identified high-importance nodes, we require methods to
identify when those nodes are compromised. We describe how Quantum
Support Vector Machine (QSVM) classifiers can be used to detect
malicious behavior in quantum networks. In particular, we describe the
detection of entanglement attacks in quantum repeaters. We show that
our initial assessment approach can be complemented by QSVM
classifiers to identify and report anomalous situations related to
malicious manipulation of entanglement swapping. Finally, we explore
the potential complexity benefits of our quantum approach compared
with classical and probabilistic methods. We also release all the
simulation code in this Github repository.

*Keywords:* Quantum Networks, Game Theory, Shapley Values, Network Security,
Quantum Graph Analytics, Cybersecurity, Quantum Machine Learning,
Quantum Support Vector Machine, Entanglement Attacks.


*Version:* April 21, 2026

### Extended Code

All the code related to our work is available in <a href="https://github.com/iain-burge/quantum_st-attack/tree/main/extended-work">this repository</a>.

### Experimental Results

Consider the following network:

![](img/extended-network.png?raw=true)

By executing <a href="https://github.com/iain-burge/quantum_st-attack/blob/main/extended-work/QuantumSTConnectivity.py">extended-work/QuantumSTConnectivity.py</a>, we obtain the results shown below:

````{verbatim}
==========================================
Node r[0]
Construct Circuit -  20:48:23
Evolve State      -  20:54:02
Display Results   -  22:08:38

    True Value:  0.0833
    Quantum:     0.0888  (Error=0.00555)
    Monte Carlo: 0.1749  (Error=0.09167)

==========================================
Node r[1]
Construct Circuit -  22:08:38
Evolve State      -  22:14:24
Display Results   -  23:08:27

    True Value:  0.03334
    Quantum:     0.03746  (Error=0.00412)
    Monte Carlo: 0.06251  (Error=0.02917)

==========================================
Node r[2]
Construct Circuit -  23:08:27
Evolve State      -  23:14:12
Display Results   -  01:34:54

    True Value:  0.28334
    Quantum:     0.29200  (Error=0.00866)
    Monte Carlo: 0.35834  (Error=0.07500)

==========================================
Node r[3]
Construct Circuit -  01:34:54
Evolve State      -  01:40:40
Display Results   -  03:17:57

    True Value:  0.03334
    Quantum:     0.03746  (Error=0.00412)
    Monte Carlo: 0.06251  (Error=0.02917)

==========================================
Node r[4]
Construct Circuit -  03:17:57
Evolve State      -  03:23:50
Display Results   -  05:03:27

    True Value:  0.28334
    Quantum:     0.29200  (Error=0.00866)
    Monte Carlo: 0.35834  (Error=0.07500)

==========================================
Node r[5]
Construct Circuit -  05:03:27
Evolve State      -  05:09:09
Display Results   -  06:50:23

    True Value:  0.00000
    Quantum:     0.00000  (Error=0.00000)
    Monte Carlo: 0.0      (Error=0.00000)

==========================================
Node r[6]
Construct Circuit -  06:50:23
Evolve State      -  06:56:10
Display Results   -  08:16:41

    True Value:  0.28334
    Quantum:     0.29200  (Error=0.00866)
    Monte Carlo: 0.35834  (Error=0.07500)
==========================================
Node r[7]
Construct Circuit -  08:20:23
Evolve State      -  08:26:10
Display Results   -  10:23:12

    True Value:  0.00000
    Quantum:     0.00000  (Error=0.00000)
    Monte Carlo: 0.0      (Error=0.00000)
==========================================

````


where the *Quantum* output is the result of Shapley value approximation
and the *Monte Carlo* output is the result of random sampling with the same
amount of samples used by the quantum approach.

The previous execution run provides the Shapley values depicted next:

![](img/extended-results.png?raw=true)

Shapley values are based on how often subsets of nodes maintain
*st*-connectivity, for example, the following subgraph indicates
that $r_3$ decides if nodes $s$ and $t$ are connected:

![](img/extended-coallition.png?raw=true)


Later on, we can assume an adversary perpetrating malicious entanglement to disrup the swapping services of some repeaters of the quantum network (cf. below figure). Using our cooperative game approach to approximating the importance of nodes that maintain a target connection, we place a classifier in critical nodes to leverage proactive defense triggered by pattern detection of malicious activities.

![](img/QSVM-Detection.png?raw=true)

The QSVM patter detection approach is simulated in <a href="https://github.com/iain-burge/quantum_st-attack/blob/main/extended-work/QSVM-Simulation.py">extended-work/QSVM-Simulation.py</a>, which provides the following confusion matrix:

![](img/QSVM-Detection_confMat.png?raw=true)

The aforementioned confusion matrix shows the classifications of a balanced random dataset. The top left quadrant represents true valid state classification; the top right quadrant is a false valid state classification; the bottom left is false malicious state classification; and the bottom right represents a true malicious state classification.

## References

If using this code for research purposes, please cite:

Iain Burge, Michel Barbeau and Joaquin Garcia-Alfaro. Identifying Vulnerable Nodes and Detecting Malicious Entanglement Patterns to Handle *st*-Connectivity Attacks in Quantum Networks, *to appear*, 2026.

```
@techreport{burge-barbeau-alfaro2026st-attack,
  title={{Identifying Vulnerable Nodes and Detecting Malicious Entanglement Patterns to Handle \emph{st}-Connectivity Attacks in Quantum Networks}},
  author={Burge, Iain and Barbeau, Michel and Garcia-Alfaro, Joaquin},
  year={2026},
  institution = {SAMOVAR, Télécom SudParis, Institut Polytechnique de Paris, 91120 Palaiseau, France},
  eprint={xxxx.yyyy},
  archivePrefix={arXiv},
  primaryClass={quant-ph},
  url={https://arxiv.org/abs/XXXX.XXX},
}
```


