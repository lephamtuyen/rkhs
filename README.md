# rkhs
Importance Sampling Policy Gradient Algorithms in Reproducing Kernel Hilbert Space

# Abstract
Modeling policies in Reproducing Kernel Hilbert Space (RKHS) offers a very flexible and powerful new family of policy gradient algorithms called RKHS policy gradient algorithms. They are designed to optimize over a space of very high or infinite dimensional policies. As a matter of fact, they are known to suffer from a large variance problem. This critical issue comes from the fact that updating the current policy is based on a functional gradient that does not exploit all old episodes sampled by previous policies. In this paper, we introduce a generalized RKHS policy gradient algorithm that integrates the following important ideas: i) policy modeling in RKHS; ii) normalized importance sampling, which helps reduce the estimation variance by reusing previously sampled episodes in a principled way; and iii) regularization terms, which avoid updating the policy too over-fit to sampled data. In the experiment section, we provide an analysis of the proposed algorithms through bench-marking domains. The experiment results show that the proposed algorithm can still enjoy a powerful policy modeling in RKHS and achieve more data-efficiency.

# Code
 + Matlab 2014b

# Paper
https://link.springer.com/article/10.1007/s10462-017-9579-x
