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

__version__ = "0.3.2"

import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import torch
from torch import hub

from ptlflow.models.base_model.base_model import BaseModel
from ptlflow.models.ccmr.ccmr import CCMR, CCMRPlus
from ptlflow.models.craft.craft import CRAFT
from ptlflow.models.csflow.csflow import CSFlow
from ptlflow.models.dicl.dicl import DICL
from ptlflow.models.dip.dip import DIP
from ptlflow.models.fastflownet.fastflownet import FastFlowNet
from ptlflow.models.flow1d.flow1d import Flow1D
from ptlflow.models.flowformer.flowformer import FlowFormer
from ptlflow.models.flowformerplusplus.flowformerplusplus import FlowFormerPlusPlus
from ptlflow.models.flownet.flownet2 import FlowNet2
from ptlflow.models.flownet.flownetc import FlowNetC
from ptlflow.models.flownet.flownetcs import FlowNetCS
from ptlflow.models.flownet.flownetcss import FlowNetCSS
from ptlflow.models.flownet.flownets import FlowNetS
from ptlflow.models.flownet.flownetsd import FlowNetSD
from ptlflow.models.gma.gma import GMA
from ptlflow.models.gmflow.gmflow import GMFlow, GMFlowWithRefinement
from ptlflow.models.gmflownet.gmflownet import GMFlowNet, GMFlowNetMix
from ptlflow.models.hd3.hd3 import HD3, HD3Context
from ptlflow.models.irr.pwcnet import IRRPWCNet
from ptlflow.models.irr.pwcnet_irr import IRRPWCNetIRR
from ptlflow.models.irr.irr_pwc import IRRPWC
from ptlflow.models.lcv.lcv_raft import LCV_RAFT, LCV_RAFTSmall
from ptlflow.models.liteflownet.liteflownet import LiteFlowNet
from ptlflow.models.liteflownet.liteflownet3 import (
    LiteFlowNet3,
    LiteFlowNet3PseudoReg,
    LiteFlowNet3S,
    LiteFlowNet3SPseudoReg,
)
from ptlflow.models.liteflownet.liteflownet2 import LiteFlowNet2, LiteFlowNet2PseudoReg
from ptlflow.models.llaflow.llaflow import LLAFlow, LLAFlowRAFT
from ptlflow.models.maskflownet.maskflownet import MaskFlownet, MaskFlownet_S
from ptlflow.models.matchflow.matchflow import MatchFlow, MatchFlowRAFT
from ptlflow.models.memflow.memflow import MemFlow, MemFlowT
from ptlflow.models.ms_raft_plus.ms_raft_plus import MSRAFTPlus
from ptlflow.models.neuflow.neuflow import NeuFlow
from ptlflow.models.neuflow2.neuflow2 import NeuFlow2
from ptlflow.models.pwcnet.pwcnet import PWCNet, PWCDCNet
from ptlflow.models.raft.raft import RAFT, RAFTSmall
from ptlflow.models.rapidflow.rapidflow import (
    RAPIDFlow,
    RAPIDFlow_it1,
    RAPIDFlow_it2,
    RAPIDFlow_it3,
    RAPIDFlow_it6,
    RAPIDFlow_it12,
)
from ptlflow.models.rpknet.rpknet import RPKNet
from ptlflow.models.sea_raft.sea_raft import SEARAFT, SEARAFT_S, SEARAFT_M, SEARAFT_L
from ptlflow.models.scopeflow.irr_pwc_v2 import ScopeFlow
from ptlflow.models.separableflow.separableflow import SeparableFlow
from ptlflow.models.skflow.skflow import SKFlow
from ptlflow.models.splatflow.splatflow import SplatFlow
from ptlflow.models.starflow.starflow import StarFlow
from ptlflow.models.unimatch.unimatch import (
    UniMatch,
    UniMatchScale2,
    UniMatchScale2With6Refinements,
)
from ptlflow.models.vcn.vcn import VCN, VCNSmall
from ptlflow.models.videoflow.videoflow_bof import VideoFlowBOF
from ptlflow.models.videoflow.videoflow_mof import VideoFlowMOF
from ptlflow.utils.utils import config_logging

try:
    from ptlflow.models.scv.scv import SCVEighth, SCVQuarter
except ImportError as e:
    print(e)
    SCVEighth = None
    SCVQuarter = None

config_logging()


models_dict = {
    "ccmr": CCMR,
    "ccmr+": CCMRPlus,
    "craft": CRAFT,
    "csflow": CSFlow,
    "dicl": DICL,
    "dip": DIP,
    "fastflownet": FastFlowNet,
    "flow1d": Flow1D,
    "flowformer": FlowFormer,
    "flowformer++": FlowFormerPlusPlus,
    "flownet2": FlowNet2,
    "flownetc": FlowNetC,
    "flownetcs": FlowNetCS,
    "flownetcss": FlowNetCSS,
    "flownets": FlowNetS,
    "flownetsd": FlowNetSD,
    "gma": GMA,
    "gmflow": GMFlow,
    "gmflow_refine": GMFlowWithRefinement,
    "gmflow+": UniMatch,
    "gmflow+_sc2": UniMatchScale2,
    "gmflow+_sc2_refine6": UniMatchScale2With6Refinements,
    "gmflownet": GMFlowNet,
    "gmflownet_mix": GMFlowNetMix,
    "hd3": HD3,
    "hd3_ctxt": HD3Context,
    "irr_pwc": IRRPWC,
    "irr_pwcnet": IRRPWCNet,
    "irr_pwcnet_irr": IRRPWCNetIRR,
    "lcv_raft": LCV_RAFT,
    "lcv_raft_small": LCV_RAFTSmall,
    "liteflownet": LiteFlowNet,
    "liteflownet2": LiteFlowNet2,
    "liteflownet2_pseudoreg": LiteFlowNet2PseudoReg,
    "liteflownet3": LiteFlowNet3,
    "liteflownet3_pseudoreg": LiteFlowNet3PseudoReg,
    "liteflownet3s": LiteFlowNet3S,
    "liteflownet3s_pseudoreg": LiteFlowNet3SPseudoReg,
    "llaflow": LLAFlow,
    "llaflow_raft": LLAFlowRAFT,
    "maskflownet": MaskFlownet,
    "maskflownet_s": MaskFlownet_S,
    "matchflow": MatchFlow,
    "matchflow_raft": MatchFlowRAFT,
    "memflow": MemFlow,
    "memflow_t": MemFlowT,
    "ms_raft+": MSRAFTPlus,
    "neuflow": NeuFlow,
    "neuflow2": NeuFlow2,
    "pwcnet": PWCDCNet,
    "pwcnet_nodc": PWCNet,
    "raft": RAFT,
    "raft_small": RAFTSmall,
    "rapidflow": RAPIDFlow,
    "rapidflow_it1": RAPIDFlow_it1,
    "rapidflow_it2": RAPIDFlow_it2,
    "rapidflow_it3": RAPIDFlow_it3,
    "rapidflow_it6": RAPIDFlow_it6,
    "rapidflow_it12": RAPIDFlow_it12,
    "rpknet": RPKNet,
    "sea_raft": SEARAFT,
    "sea_raft_s": SEARAFT_S,
    "sea_raft_m": SEARAFT_M,
    "sea_raft_l": SEARAFT_L,
    "scopeflow": ScopeFlow,
    "scv4": SCVQuarter,
    "scv8": SCVEighth,
    "separableflow": SeparableFlow,
    "skflow": SKFlow,
    "splatflow": SplatFlow,
    "starflow": StarFlow,
    "unimatch": UniMatch,
    "unimatch_sc2": UniMatchScale2,
    "unimatch_sc2_refine6": UniMatchScale2With6Refinements,
    "vcn": VCN,
    "vcn_small": VCNSmall,
    "videoflow_bof": VideoFlowBOF,
    "videoflow_mof": VideoFlowMOF,
}


def download_scripts(destination_dir: Path = Path("ptlflow_scripts")) -> None:
    """Download the main scripts and configs to start working with PTLFlow."""
    github_url = "https://raw.githubusercontent.com/hmorimitsu/ptlflow/main/"
    script_names = [
        "datasets.yml",
        "infer.py",
        "model_benchmark.py",
        "test.py",
        "train.py",
        "validate.py",
    ]

    destination_dir.mkdir(parents=True, exist_ok=True)

    for sname in script_names:
        script_url = github_url + sname
        data = requests.get(script_url)
        if data.status_code == 200:
            with open(destination_dir / sname, "wb") as f:
                f.write(data.content)
        else:
            logging.warning("Script %s was not found.", script_url)

    logging.info("Downloaded scripts to %s.", str(destination_dir))


def get_model(
    model_name: str,
    pretrained_ckpt: Optional[str] = None,
    args: Optional[Namespace] = None,
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

    if (
        pretrained_ckpt is None
        and args is not None
        and args.pretrained_ckpt is not None
    ):
        pretrained_ckpt = args.pretrained_ckpt

    if pretrained_ckpt is not None:
        ckpt = load_checkpoint(pretrained_ckpt, model_ref, model_name)

        state_dict = ckpt["state_dict"]
        if "hyper_parameters" in ckpt:
            if "train_size" in ckpt["hyper_parameters"]:
                model.train_size = ckpt["hyper_parameters"]["train_size"]
            if "train_avg_length" in ckpt["hyper_parameters"]:
                model.train_avg_length = ckpt["hyper_parameters"]["train_avg_length"]
            if "extra_params" in ckpt["hyper_parameters"]:
                extra_params = ckpt["hyper_parameters"]["extra_params"]
                for name, value in extra_params.items():
                    model.add_extra_param(name, value)
        model.load_state_dict(state_dict)

    return model


def get_model_reference(model_name: str) -> BaseModel:
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
        raise ValueError(
            f'Unknown model name: {model_name}. Choose from [{", ".join(models_dict.keys())}]'
        )


def get_trainable_model_names() -> List[str]:
    """Return a list of model names that are able to be trained.

    This function return the names of the model that have a loss function defined.

    Returns
    -------
    List[str]
        The list of the model names that can be trained.
    """
    return [
        mname for mname in models_dict.keys() if get_model(mname).loss_fn is not None
    ]


def load_checkpoint(
    pretrained_ckpt: str, model_ref: BaseModel, model_name: str
) -> Dict[str, Any]:
    """Try to load the checkpoint specified in pretrained_ckpt.

    Parameters
    ----------
    pretrained_ckpt : str
        Path to a local file or name of a pretrained checkpoint.
    model_ref : BaseModel
        A reference to the model class. See the function get_model_reference() for more details.
    model_name : str
        A string representing the name of the model, just for debugging purposes.

    Returns
    -------
    Dict[str, Any]
        A dictionary of the loaded checkpoint. The output of torch.load().

    See Also
    --------
    get_model_reference : To get a reference to the class of a model.
    """
    if Path(pretrained_ckpt).exists():
        ckpt_path = pretrained_ckpt
    elif hasattr(model_ref, "pretrained_checkpoints"):
        ckpt_path = model_ref.pretrained_checkpoints.get(pretrained_ckpt)
        if ckpt_path is None:
            raise ValueError(
                f"Invalid checkpoint name {pretrained_ckpt}. "
                f'Choose one from {{{",".join(model_ref.pretrained_checkpoints.keys())}}}'
            )
    else:
        raise ValueError(
            f"Cannot find checkpoint {pretrained_ckpt} for model {model_name}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    else:
        model_dir = Path(hub.get_dir()) / "ptlflow" / "checkpoints"
        ckpt = hub.load_state_dict_from_url(
            ckpt_path,
            model_dir=model_dir,
            map_location=torch.device(device),
            check_hash=True,
        )
    return ckpt
