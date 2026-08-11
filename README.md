Supplementary Material to: Identifying Vulnerable Nodes and Detecting Malicious Entanglement Patterns to Handle *st*-Connectivity Attacks in Quantum Networks
===

<a href="https://github.com/iain-burge/iain-burge">Iain Burge, Institut Polytechnique de Paris, France.</a>

<a href="https://carleton.ca/scs/people/michel-barbeau/">Michel Barbeau, Carleton University, School of Computer Science, Canada.</a>

<a href="http://j.mp/jgalfaro">Joaquin Garcia-Alfaro, Institut Polytechnique de Paris, France.</a>

## Abstract

Several problems in distributed system security naturally map to
graphs. The concept of centrality assesses the importance of nodes in
a graph. It is used in various applications. Cooperative game theory
has also been used to create nuanced and flexible notions of node
centrality. However, the approach is often computationally complex to
implement in classical settings. Our first contribution describes a
quantum approach to approximating the importance of quantum nodes that
support a target connection in a quantum network. We detail a method
for quickly identifying high-importance nodes that adversaries can
target. The approximation method relies on quantum subroutines for
evaluating *st*-connectivity, approximating Shapley values, and
finding the maximum of a list. We consider a malicious actor targeting
a subset of nodes to disrupt the system functionality. Our method
identifies the nodes that are most important in keeping nodes *s* and
*t* connected. Once we have identified high-importance nodes, we
require methods to identify when those nodes are compromised. Our
second contribution describes how Quantum Support Vector Machine
(QSVM) classifiers can be used to detect malicious behavior in quantum
networks. In particular, we describe the detection of entanglement
attacks in quantum repeaters. We show that our initial assessment
approach can be complemented by QSVM classifiers to identify and alert
when anomalous situations related to malicious manipulation of
entanglement swapping occur. Finally, we explore the potential
complexity benefits of our quantum approach compared with classical
and probabilistic methods. We also release all the simulation code and
artifacts associated to the work in this Github repository, to foster
reproducibility and further research on the topic.

*Keywords:* Quantum Networks, Game Theory, Shapley Values, Network Security,
Quantum Graph Analytics, Cybersecurity, Quantum Machine Learning,
Quantum Support Vector Machine, Entanglement Attacks.


*Version:* August 15, 2026

### Full manuscript

A pre-print version of our work is available at
<a href="https://doi.org/10.48550/arXiv.2502.00446">https://doi.org/10.48550/arXiv.2502.00446.

### Full Release of Code and Artifacts

All the code related to our work, including explanations to install and reproduce the resulst, is available in <a href="https://github.com/iain-burge/quantum_st-attack/tree/main/extended-work">this repository</a>. Some additional explanations about the obtained results follows.

### 1. Quantum approach to approximating the importance of quantum nodes: Experimental Results

Consider the following network:

![](img/extended-network.png?raw=true)

By executing <a href="https://github.com/iain-burge/quantum_st-attack/blob/main/extended-work/QuantumSTConnectivity.py">extended-work/QuantumSTConnectivity.py</a>, we obtain the results shown below:

````{verbatim}
==========================================
Node r[0]
Construct Circuit -  20:48:23
Evolve State      -  20:54:02
Display Results   -  22:08:38

    True Value:  0.08333
    Quantum:     0.08427  (Error=0.00093)
    Monte Carlo: 0.09375  (Error=0.01042)

==========================================
Node r[1]
Construct Circuit -  22:08:38
Evolve State      -  22:14:24
Display Results   -  23:08:27

    True Value:  0.03333
    Quantum:     0.03806  (Error=0.00473)
    Monte Carlo: 0.0      (Error=0.03333)

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


### 2. Quantum Support Vector Machine (QSVM) to detect malicious behavior: Experimental Results

The above figures shows that nodes $r_2,r_4,$ and $r_6$ are valuable
targets. To detect an entanglement attack originating from $r_4$, we
perform the following steps.

Node $r_2$ constructs two identical two-qubit quantum states denoted
as} $A_1,B_1$ for the first pair, and $A_2,B_2$ for the second pair,
where $A_k$ and $B_k$ are entangled. $r_2$ sends both pairs, $A_1,B_1$
and $A_2,B_2$, to $r_6$ via $r_4$. Based on our adversarial model, if
$r_4$ is compromised, it is possible for $r_6$ to receive,
$A_1B_1A_2B_2$, $A_1C_1A_2B_2$, $A_1B_1A_2C_2$, or $A_1C_1A_2C_2$,
where $C_k$ is an arbitrary qubit not entangled with $A_k$. Finally,
$r_6$ uses a quantum SVM trained with synthetic data that
distinguishes the expected state $A_1B_1A_2B_2$, from malicious states
$A_1C_1A_2B_2$, $A_1B_1A_2C_2$, or $A_1C_1A_2C_2$.


To experimentally validate the theoretical work, we ran our detection
method on a series of datasets corresponding to the target attack
described in Section~\ref{sec:detectingEntAttack}. The attack is
implemented using <a
href="https://github.com/omnetpp/omnetpp/blob/omnetpp-6.0.3/">OMNeT++
6.0.3</a> and <a href="https://github.com/sfc-aqua/quisp">QuISP
0.3</a>. All the simulation code and resulting artifacts are released
<a
href="https://github.com/iain-burge/quantum_st-attack/tree/main/extended-work/quisp-exported-datasets">in
this folder</a>. See the following figure for a representative
screenshot with the experimental environment.

![](img/expEnvironment.png?raw=true)

To test the code, we produced $512$ legitimate scenarios and $512$
malicious scenarios. In the legitimate scenarios, $A_1B_1A_2B_2$ were
received. In the malicious scenarios, $A_1C_1A_2B_2$, $A_1B_1A_2C_2$,
or $A_1C_1A_2C_2$, were received with probabilities $40\%$, $40\%$ or
$20\%$ respectively.

![](img/results.png?raw=true)



## References

If using this code for research purposes, please cite:

Iain Burge, Michel Barbeau and Joaquin Garcia-Alfaro. Identifying vulnerable nodes and detecting malicious entanglement patterns to handle st-connectivity attacks in quantum networks, *to appear*, 2026.

```
@techreport{burge-barbeau-alfaro2026st-attack,
  title={Identifying vulnerable nodes and detecting malicious entanglement patterns to handle st-connectivity attacks in quantum networks},
  author={Burge, Iain and Barbeau, Michel and Garcia-Alfaro, Joaquin},
  year={2026},
  archivePrefix={arXiv},
  primaryClass={quant-ph},
  url={https://arxiv.org/abs/2502.00446v3},
}
```


