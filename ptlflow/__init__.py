"""Provide useful functions for using PTLFlow."""

# =============================================================================
# Copyright 2021 Henrique Morimitsu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

__version__ = '0.2.1'

import logging
from argparse import Namespace
from pathlib import Path
from typing import List, Optional

import requests
import torch
from torch import hub

from ptlflow.models.base_model.base_model import BaseModel
from ptlflow.models.flownet.flownet2 import FlowNet2
from ptlflow.models.flownet.flownetc import FlowNetC
from ptlflow.models.flownet.flownetcs import FlowNetCS
from ptlflow.models.flownet.flownetcss import FlowNetCSS
from ptlflow.models.flownet.flownets import FlowNetS
from ptlflow.models.flownet.flownetsd import FlowNetSD
from ptlflow.models.hd3.hd3 import HD3, HD3Context
from ptlflow.models.irr.pwcnet import IRRPWCNet
from ptlflow.models.irr.pwcnet_irr import IRRPWCNetIRR
from ptlflow.models.irr.irr_pwc import IRRPWC
from ptlflow.models.liteflownet.liteflownet import LiteFlowNet
from ptlflow.models.liteflownet.liteflownet3 import (
    LiteFlowNet3, LiteFlowNet3PseudoReg, LiteFlowNet3S, LiteFlowNet3SPseudoReg)
from ptlflow.models.liteflownet.liteflownet2 import LiteFlowNet2, LiteFlowNet2PseudoReg
from ptlflow.models.pwcnet.pwcnet import PWCNet, PWCDCNet
from ptlflow.models.raft.raft import RAFT, RAFTSmall
from ptlflow.models.scopeflow.irr_pwc_v2 import ScopeFlow
from ptlflow.models.starflow.starflow import StarFlow
from ptlflow.models.vcn.vcn import VCN, VCNSmall
from ptlflow.utils.utils import config_logging

config_logging()


models_dict = {
    'flownet2': FlowNet2,
    'flownetc': FlowNetC,
    'flownetcs': FlowNetCS,
    'flownetcss': FlowNetCSS,
    'flownets': FlowNetS,
    'flownetsd': FlowNetSD,
    'hd3': HD3,
    'hd3_ctxt': HD3Context,
    'irr_pwc': IRRPWC,
    'irr_pwcnet': IRRPWCNet,
    'irr_pwcnet_irr': IRRPWCNetIRR,
    'liteflownet': LiteFlowNet,
    'liteflownet2': LiteFlowNet2,
    'liteflownet2_pseudoreg': LiteFlowNet2PseudoReg,
    'liteflownet3': LiteFlowNet3,
    'liteflownet3_pseudoreg': LiteFlowNet3PseudoReg,
    'liteflownet3s': LiteFlowNet3S,
    'liteflownet3s_pseudoreg': LiteFlowNet3SPseudoReg,
    'pwcnet': PWCNet,
    'pwcdcnet': PWCDCNet,
    'raft': RAFT,
    'raft_small': RAFTSmall,
    'scopeflow': ScopeFlow,
    'starflow': StarFlow,
    'vcn': VCN,
    'vcn_small': VCNSmall,
}


def download_scripts(
    destination_dir: Path = Path('ptlflow_scripts')
) -> None:
    """Download the main scripts and configs to start working with PTLFlow."""
    github_url = 'https://raw.githubusercontent.com/hmorimitsu/ptlflow/main/'
    script_names = [
        'datasets.yml',
        'infer.py',
        'train.py',
        'validate.py'
    ]

    destination_dir.mkdir(parents=True, exist_ok=True)

    for sname in script_names:
        script_url = github_url + sname
        data = requests.get(script_url)
        if data.status_code == 200:
            with open(destination_dir / sname, 'wb') as f:
                f.write(data.content)
        else:
            logging.warning('Script %s was not found.', script_url)

    logging.info('Downloaded scripts to %s.', str(destination_dir))


def get_model(
    model_name: str,
    pretrained_ckpt: Optional[str] = None,
    args: Optional[Namespace] = None
) -> BaseModel:
    """Return an instance of a chosen model.

    The instance can have configured by he arguments, and load some existing pretrained weights.

    Note that this is different from get_model_reference(), which returns a reference to the model class. The instance,
    returned by this function, is a class already instantiated. Therefore, the return of this function is equivalent to
    "return get_model_reference()()", which looks confusing. This can be rewritten as
    "model_ref = get_model_reference(); return model_ref()".

    Parameters
    ----------
    model_name : str
        Name of the model to get an instance of.
    pretrained_ckpt : Optional[str], optional
        Name of the pretrained weight to load or a path to a local checkpoint file.
    args : Optional[Namespace], optional
        Some arguments that ill be provided to the model.

    Returns
    -------
    BaseModel
        The instance of the chosen model.

    Raises
    ------
    ValueError
        If the given checkpoint name is not a valid choice.
    ValueError
        If a checkpoint name is given, but the model does not have any pretrained weights available.

    See Also
    --------
    get_model_reference : To get a reference to the class of a model.
    """
    model_ref = get_model_reference(model_name)
    if args is None:
        parser = model_ref.add_model_specific_args()
        args = parser.parse_args([])
    model = model_ref(args)

    if pretrained_ckpt is None and args is not None and args.pretrained_ckpt is not None:
        pretrained_ckpt = args.pretrained_ckpt

    if pretrained_ckpt is not None:
        if Path(pretrained_ckpt).exists():
            ckpt_path = pretrained_ckpt
        elif hasattr(model_ref, 'pretrained_checkpoints'):
            ckpt_path = model_ref.pretrained_checkpoints.get(pretrained_ckpt)
            if ckpt_path is None:
                raise ValueError(
                    f'Invalid checkpoint name {pretrained_ckpt}. '
                    f'Choose one from {{{",".join(model.pretrained_checkpoints.keys())}}}')
        else:
            raise ValueError(f'Cannot find checkpoint {pretrained_ckpt} for model {model_name}')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if Path(ckpt_path).exists():
            ckpt = torch.load(ckpt_path, map_location=torch.device(device))
        else:
            model_dir = Path(hub.get_dir()) / 'ptlflow' / 'checkpoints'
            ckpt = hub.load_state_dict_from_url(
                ckpt_path, model_dir=model_dir, map_location=torch.device(device), check_hash=True)

        state_dict = ckpt['state_dict']
        model.load_state_dict(state_dict)
    return model


def get_model_reference(
    model_name: str
) -> BaseModel:
    """Return a reference to the class of a chosen model.

    Note that this is different from get_model(), which returns an instance of a model. The reference, returned by this
    function, is a class before instantiation. Therefore, the return of this function can be used to instantiate a model as
    "model_ref = get_model_reference(); model_instance = model_ref()".

    Parameters
    ----------
    model_name : str
        Name of the model to get a reference of.

    Returns
    -------
    BaseModel
        A reference to the chosen model.

    Raises
    ------
    ValueError
        If the given name is not a valid choice.

    See Also
    --------
    get_model : To get an instance of a model.
    """
    try:
        return models_dict[model_name]
    except KeyError:
        raise ValueError(f'Unknown model name: {model_name}. Choose from [{", ".join(models_dict.keys())}]')


def get_trainable_model_names() -> List[str]:
    """Return a list of model names that are able to be trained.
    
    This function return the names of the model that have a loss function defined.

    Returns
    =======
    List[str]
        The list of the model names that can be trained.
    """
    return [mname for mname in models_dict.keys() if get_model(mname).loss_fn is not None]
