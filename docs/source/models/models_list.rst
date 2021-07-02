========================
List of available models
========================

Below is a list and a brief explanation about the models currently available on PTLFlow.

.. _external-models:

External models
===============

There are two types of implementations for the models. "*Internal*" models are implemented
for PTLFlow and they should work with all stages of the platform. "*External*" models, on the other hand,
are mostly taken from external sources and lightly adapted for being compatible with the platform.
External models should work for inference and validation inside PTLFlow. However, the training
is not guaranteed to work. External models are placed in a separated folder called ``models/external``.
They are also identified by the ``ext_`` prefix in their names.

  Obs: at the moment, only external models are available.

List of models
==============

Flownet
-------

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/external/flownet>`__

- Papers:

  - **FlowNet: Learning Optical Flow with Convolutional Networks** - `https://arxiv.org/abs/1504.06852 <https://arxiv.org/abs/1504.06852>`_

  - **FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks** - `https://arxiv.org/abs/1612.01925 <https://arxiv.org/abs/1612.01925>`_

- Reference code: `https://github.com/NVIDIA/flownet2-pytorch <https://github.com/NVIDIA/flownet2-pytorch>`_

- Model names: ``ext_flownets``, ``ext_flownetc``, ``ext_flownet2``, ``ext_flownetcs``, ``ext_flownetcss``, ``ext_flownetsd``

HD3
---

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/external/hd3>`__

- Paper: **Hierarchical Discrete Distribution Decomposition for Match Density Estimation** - `https://arxiv.org/abs/1812.06264 <https://arxiv.org/abs/1812.06264>`_

- Reference code: `https://github.com/ucbdrive/hd3 <https://github.com/ucbdrive/hd3>`_

- Model names: ``ext_hd3``, ``ext_hd3_ctxt``


IRR
---

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/external/irr>`__

- Paper: **Iterative Residual Refinement for Joint Optical Flow and Occlusion Estimation** - `https://arxiv.org/abs/1904.05290 <https://arxiv.org/abs/1904.05290>`_

- Reference code: `https://github.com/visinf/irr <https://github.com/visinf/irr>`_

- Model names: ``ext_irr_pwc``, ``ext_irr_pwcnet``, ``ext_irr_pwcnet_irr``

LiteFlowNet
-----------

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/external/liteflownet>`__

- Paper: **LiteFlowNet: A Lightweight Convolutional Neural Network for Optical Flow Estimation** - `https://arxiv.org/abs/1805.07036 <https://arxiv.org/abs/1805.07036>`_

- Reference code: `https://github.com/twhui/LiteFlowNet <https://github.com/twhui/LiteFlowNet>`__

- Model name: ``ext_liteflownet``

LiteFlowNet2
------------

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/external/liteflownet>`__

- Paper: **A Lightweight Optical Flow CNN - Revisiting Data Fidelity and Regularization** - `https://ieeexplore.ieee.org/document/9018073 <https://ieeexplore.ieee.org/document/9018073>`_

- Reference code: `https://github.com/twhui/LiteFlowNet2 <https://github.com/twhui/LiteFlowNet2>`__

- Model names: ``ext_liteflownet2``, ``ext_liteflownet2_pseudoreg``

LiteFlowNet3
------------

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/external/liteflownet>`__

- Paper: **LiteFlowNet3: Resolving Correspondence Ambiguity for More Accurate Optical Flow Estimation** - `https://arxiv.org/abs/2007.09319 <https://arxiv.org/abs/2007.09319>`_

- Reference code: `https://github.com/twhui/LiteFlowNet <https://github.com/twhui/LiteFlowNet3>`__

- Model names: ``ext_liteflownet3``, ``ext_liteflownet3_pseudoreg``, ``ext_liteflownet3s``, ``ext_liteflownet3s_pseudoreg``

PWCNet
------

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/external/pwcnet>`__

- Paper: **PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume** - `https://arxiv.org/abs/1709.02371 <https://arxiv.org/abs/1709.02371>`_

- Reference code: `https://github.com/NVlabs/PWC-Net <https://github.com/NVlabs/PWC-Net>`_

- Model names: ``ext_pwcnet``, ``ext_pwcdcnet``

RAFT
----

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/external/raft>`__

- Paper: **RAFT: Recurrent All-Pairs Field Transforms for Optical Flow** - `https://arxiv.org/abs/2003.12039 <https://arxiv.org/abs/2003.12039>`_

- Reference code: `https://github.com/princeton-vl/RAFT <https://github.com/princeton-vl/RAFT>`_

- Model names: ``ext_raft``, ``ext_raft_small``

ScopeFlow
---------

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/external/scopeflow>`__

- Paper: **ScopeFlow: Dynamic Scene Scoping for Optical Flow** - `https://arxiv.org/abs/2002.10770 <https://arxiv.org/abs/2002.10770>`_

- Reference code: `https://github.com/avirambh/ScopeFlow <https://github.com/avirambh/ScopeFlow>`_

- Model names: ``ext_scopeflow``

VCN
---

`[source code] <https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/external/vcn>`__

- Paper: **Volumetric Correspondence Networks for Optical Flow** - `https://papers.nips.cc/paper/2019/file/bbf94b34eb32268ada57a3be5062fe7d-Paper.pdf <https://papers.nips.cc/paper/2019/file/bbf94b34eb32268ada57a3be5062fe7d-Paper.pdf>`_

- Reference code: `https://github.com/gengshan-y/VCN <https://github.com/gengshan-y/VCN>`_

- Model names: ``ext_vcn``, ``ext_vcn_small``