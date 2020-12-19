# Tri_GP_finder

This is an algorithm based on a generic algorithm and a reinforced learning. 
For a given positive integer n, it tries to find an integer partition of the form

n = T_{k_1} + T_{k_2} + ... + T_{k_m} 

and

1  = 1/k_1 + 1/k_2 + ... + 1/k_m

where T_k is k-th triangular number. 

For example, 
6 = 3+3
and
1 = 1/2 + 1/2.

For a running example on n=1728 and n=1729, see the text file.

We give a list of such partitions of n from n=500 to n=1004. 
There is no such a partition for n=500, 503, 518, 529, 570, 589, 644.
The included agent (agent1) fails to find such a partition for n=565, 653, 685, 719, 774.

In the table, the first column is a given integer n and the numbers in the other columns are parts. 


The examples in the table supports that there is such a partition for n > 644.



The policy network is trained using TD3 algorithm and a trained agent (agent1_policy) is included.
I use TD3 paper's authors code from
https://github.com/sfujim/TD3/blob/master/TD3.py


Dependency: Python 3.6 and Pytorch 1.6
