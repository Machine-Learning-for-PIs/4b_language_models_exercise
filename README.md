# Language Modelling Exercise

This exercsie will allow you to explore language modelling. We focus on the key concept of multi-head attention.
Navigate to the `src/attention_model.py`-file and implement multi-head attention [1]

``` math
\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}})\mathbf{V}
```

To make attention useful in a language modelling scenario we cannot use future information. A model without access to upcoming future inputs or words is known as causal.
Since our attention matrix is multiplied from the left we must mask out the upper triangle
excluding the main diagonal for causality.

Keep in mind that $\mathbf{Q} \in \mathbb{R}^{b,h,o,d_k}$, $\mathbf{Q} \in \mathbb{R}^{b,h,o,d_k}$ and $\mathbf{Q} \in \mathbb{R}^{b,h,o,d_v}$, with $b$ the batch size, $h$ the number of heads, $o$ the desired output dimension, $d_k$ the key dimension and finally $d_v$ as value dimension. Your code must rely on broadcasting to process the matrix operations correctly. The notation follows [1]. 

Furthermore write a function to convert the network output of vector encodings back into a string by completing the `convert` function  in `src/util.py`.


[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin:
Attention is All you Need. NIPS 2017: 5998-6008

Once you have implemented and tested your version of attention run `sbatch scripts/train.slurm` to train your model on Bender. Once converged you can generate poetry via `sbatch scripts/generate.slurm`.
Run `src/model_chat.py` to talk to your model.
