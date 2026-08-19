Supplementary Material to: Identifying Vulnerable Nodes and Detecting Malicious Entanglement Patterns to Handle *st*-Connectivity Attacks in Quantum Networks
===

<a href="https://github.com/iain-burge/iain-burge">Iain Burge, Institut Polytechnique de Paris, France.</a>

<a href="https://carleton.ca/scs/people/michel-barbeau/">Michel Barbeau, Carleton University, School of Computer Science, Canada.</a>

<a href="http://j.mp/jgalfaro">Joaquin Garcia-Alfaro, Institut Polytechnique de Paris, France.</a>

### Quantum Support Vector Machine (QSVM) to detect malicious behavior: Experimental Results

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
described in Section 2 of the <a
href="https://doi.org/10.48550/arXiv.2502.00446">following
pre-print</a>. The attack is implemented using <a
href="https://github.com/omnetpp/omnetpp/blob/omnetpp-6.0.3/">OMNeT++
6.0.3</a> and <a href="https://github.com/sfc-aqua/quisp">QuISP
0.3</a>.

All the simulation code and resulting artifacts are available in the
following <a
href="https://github.com/jgalfaro/quisp-PRE">repository</a>. A VirtualBox Virtual Machine ready to run and reproduce the experiments is available in <a href="https://filesender.renater.fr/?s=download&token=c2a65407-985f-4061-a12a-7f40db4e84b4">this link</a>. Additional instructions to extract and reproduce the experimental work is provided in the following videocapture:

[![](https://github.com/iain-burge/quantum_st-attack/raw/main/img/expEnvironment.png?raw=true)](https://www.youtube.com/watch?v=FnYSJW9GQss)

To evaluate the performance of the detection model, we produced $512$
legitimate scenarios and $512$ malicious scenarios. In the legitimate
scenarios, $A_1B_1A_2B_2$ were received. In the malicious scenarios,
$A_1C_1A_2B_2$, $A_1B_1A_2C_2$, or $A_1C_1A_2C_2$, were received with
probabilities $40\%$, $40\%$ or $20\%$ respectively.

In particular, with $\ell=2$ ($512$ datapoint training set), we found
the average of $|f|$ over the trials was $1.36\times10^{-3}$
(std $=8.83\times10^{-4}$). But, with $\ell=3$ ($8192$ datapoint
training set), we found $|f|$ to average $2.22\times10^{-4}$
(std $=1.83\times10^{-4}$). The following figure provides a confusion
matrix summarizing the obtained results. It shows that the detection
model makes the correct classification with high accuracy, in
practice.

<img src="https://github.com/iain-burge/quantum_st-attack/blob/main/img/results.png" width="45%" height="45%">

The aforementioned datasets, to validate and reproduce the confusion matrix are available in the following three CSV files:

1. <a href="https://github.com/iain-burge/quantum_st-attack/blob/main/extended-work/quisp-exported-datasets/groundtruth-teleported-states.csv">groundtruth-teleported-states.csv</a>

1. <a href="https://github.com/iain-burge/quantum_st-attack/blob/main/extended-work/quisp-exported-datasets/groundtruth-teleported-matrices.csv">groundtruth-teleported-matrices.csv</a>

1. <a href="https://github.com/iain-burge/quantum_st-attack/blob/main/extended-work/quisp-exported-datasets/attacked-teleported-matrices.csv">attacked-teleported-matrices.csv</a>

In case additional information or details are required, kindly contact the authors:

<a href="mailto:iain-james.burge@telecom-sudparis.eu">Iain Burge, Institut Polytechnique de Paris, France.</a>

<a href="mailto:barbeau@scs.carleton.ca">Michel Barbeau, Carleton University, School of Computer Science, Canada.</a>

<a href="mailto:jgalfaro@ieee.org">Joaquin Garcia-Alfaro, Institut Polytechnique de Paris, France.</a>


## References

If using this code for research purposes, please cite:

Iain Burge, Michel Barbeau and Joaquin Garcia-Alfaro. Identifying vulnerable nodes and detecting malicious entanglement patterns to handle st-connectivity attacks in quantum networks, *to appear*, 2026.

```
@misc{burge-barbeau-alfaro2026st-attack,
  title={Identifying vulnerable nodes and detecting malicious entanglement patterns to handle st-connectivity attacks in quantum networks},
  author={Burge, Iain and Barbeau, Michel and Garcia-Alfaro, Joaquin},
  year={2026},
  archivePrefix={arXiv},
  primaryClass={quant-ph},
  url={https://arxiv.org/abs/2502.00446v3},
}
```


