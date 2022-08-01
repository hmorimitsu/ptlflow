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

from pathlib import Path
from typing import Dict

import cv2 as cv
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import ptlflow
from ptlflow.data.flow_transforms import ToTensor
from ptlflow.utils import flow_utils
from ptlflow.utils.utils import InputPadder


# Results at scale_factor=0.66
reference_accuracy = {
    'craft_things_flyingchairs': 0.720,
    'craft_things_flyingthings3d': 3.317,
    'craft_things_kitti': 7.534,
    'craft_things_sintel': 0.186,
	'craft_sintel_flyingchairs': 0.680,
    'craft_sintel_flyingthings3d': 4.002,
    'craft_sintel_kitti': 1.863,
    'craft_sintel_sintel': 0.143,
	'craft_kitti_flyingchairs': 1.713,
    'craft_kitti_flyingthings3d': 19.485,
    'craft_kitti_kitti': 0.921,
    'craft_kitti_sintel': 0.238,
	'csflow_chairs_flyingchairs': 0.587,
    'csflow_chairs_flyingthings3d': 8.454,
    'csflow_chairs_kitti': 12.062,
    'csflow_chairs_sintel': 0.223,
    'csflow_things_flyingchairs': 0.843,
    'csflow_things_flyingthings3d': 3.620,
    'csflow_things_kitti': 4.639,
    'csflow_things_sintel': 0.185,
    'csflow_kitti_flyingchairs': 0.920,
    'csflow_kitti_flyingthings3d': 6.655,
    'csflow_kitti_kitti': 1.205,
    'csflow_kitti_sintel': 0.168,
    'dicl_chairs_flyingchairs': 0.675,
    'dicl_chairs_flyingthings3d': 20.257,
    'dicl_chairs_kitti': 24.210,
    'dicl_chairs_sintel': 0.373,
    'dicl_kitti_flyingchairs': 2.210,
    'dicl_kitti_flyingthings3d': 72.860,
    'dicl_kitti_kitti': 3.729,
    'dicl_kitti_sintel': 0.368,
    'dicl_sintel_flyingchairs': 1.809,
    'dicl_sintel_flyingthings3d': 8.649,
    'dicl_sintel_kitti': 7.003,
    'dicl_sintel_sintel': 0.185,
    'dicl_things_flyingchairs': 1.469,
    'dicl_things_flyingthings3d': 6.065,
    'dicl_things_kitti': 17.919,
    'dicl_things_sintel': 0.240,
    'fastflownet_chairs_flyingchairs': 1.534,
    'fastflownet_chairs_flyingthings3d': 21.064,
    'fastflownet_chairs_kitti': 11.806,
    'fastflownet_chairs_sintel': 0.455,
    'fastflownet_kitti_flyingchairs': 2.000,
    'fastflownet_kitti_flyingthings3d': 32.696,
    'fastflownet_kitti_kitti': 2.860,
    'fastflownet_kitti_sintel': 0.378,
    'fastflownet_mix_flyingchairs': 1.423,
    'fastflownet_mix_flyingthings3d': 17.350,
    'fastflownet_mix_kitti': 5.386,
    'fastflownet_mix_sintel': 0.344,
    'fastflownet_sintel_flyingchairs': 1.564,
    'fastflownet_sintel_flyingthings3d': 16.783,
    'fastflownet_sintel_kitti': 13.789,
    'fastflownet_sintel_sintel': 0.349,
    'fastflownet_things_flyingchairs': 1.648,
    'fastflownet_things_flyingthings3d': 20.497,
    'fastflownet_things_kitti': 12.205,
    'fastflownet_things_sintel': 0.434,
    'flowformer_chairs_flyingchairs': 0.558,
    'flowformer_chairs_flyingthings3d': 11.260,
    'flowformer_chairs_kitti': 13.684,
    'flowformer_chairs_sintel': 0.261,
    'flowformer_things_flyingchairs': 0.697,
    'flowformer_things_flyingthings3d': 3.104,
    'flowformer_things_kitti': 6.169,
    'flowformer_things_sintel': 0.245,
    'flowformer_sintel_flyingchairs': 0.650,
    'flowformer_sintel_flyingthings3d': 2.960,
    'flowformer_sintel_kitti': 2.377,
    'flowformer_sintel_sintel': 0.175,
    'flowformer_kitti_flyingchairs': 1.605,
    'flowformer_kitti_flyingthings3d': 17.919,
    'flowformer_kitti_kitti': 1.888,
    'flowformer_kitti_sintel': 0.325,
    'flownet2_things_flyingchairs': 1.986,
    'flownet2_things_flyingthings3d': 10.010,
    'flownet2_things_kitti': 16.391,
    'flownet2_things_sintel': 0.551,
    'flownetc_things_flyingchairs': 2.803,
    'flownetc_things_flyingthings3d': 12.762,
    'flownetc_things_kitti': 15.847,
    'flownetc_things_sintel': 1.149,
    'flownetcs_things_flyingchairs': 1.759,
    'flownetcs_things_flyingthings3d': 8.329,
    'flownetcs_things_kitti': 13.826,
    'flownetcs_things_sintel': 0.390,
    'flownetcss_things_flyingchairs': 1.637,
    'flownetcss_things_flyingthings3d': 7.818,
    'flownetcss_things_kitti': 14.002,
    'flownetcss_things_sintel': 0.316,
    'flownets_things_flyingchairs': 1.828,
    'flownets_things_flyingthings3d': 15.145,
    'flownets_things_kitti': 13.089,
    'flownets_things_sintel': 0.857,
    'flownetsd_things_flyingchairs': 1.814,
    'flownetsd_things_flyingthings3d': 34.579,
    'flownetsd_things_kitti': 22.438,
    'flownetsd_things_sintel': 0.255,
    'hd3_chairs_flyingchairs': 0.865,
    'hd3_chairs_flyingthings3d': 31.540,
    'hd3_chairs_kitti': 24.647,
    'hd3_chairs_sintel': 0.534,
    'hd3_things_flyingchairs': 1.318,
    'hd3_things_flyingthings3d': 8.197,
    'hd3_things_kitti': 12.478,
    'hd3_things_sintel': 0.282,
    'hd3_sintel_flyingchairs': 2.016,
    'hd3_sintel_flyingthings3d': 23.785,
    'hd3_sintel_kitti': 21.161,
    'hd3_sintel_sintel': 0.227,
    'hd3_kitti_flyingchairs': 1.766,
    'hd3_kitti_flyingthings3d': 166.434,
    'hd3_kitti_kitti': 1.924,
    'hd3_kitti_sintel': 0.285,
    'hd3_ctxt_chairs_flyingchairs': 0.828,
    'hd3_ctxt_chairs_flyingthings3d': 47.454,
    'hd3_ctxt_chairs_kitti': 29.977,
    'hd3_ctxt_chairs_sintel': 0.462,
    'hd3_ctxt_things_flyingchairs': 1.280,
    'hd3_ctxt_things_flyingthings3d': 9.238,
    'hd3_ctxt_things_kitti': 13.115,
    'hd3_ctxt_things_sintel': 0.249,
    'hd3_ctxt_sintel_flyingchairs': 1.896,
    'hd3_ctxt_sintel_flyingthings3d': 14.648,
    'hd3_ctxt_sintel_kitti': 14.455,
    'hd3_ctxt_sintel_sintel': 0.198,
    'hd3_ctxt_kitti_flyingchairs': 2.059,
    'hd3_ctxt_kitti_flyingthings3d': 66.693,
    'hd3_ctxt_kitti_kitti': 1.491,
    'hd3_ctxt_kitti_sintel': 0.305,
    'gma_chairs_flyingchairs': 0.533,
    'gma_chairs_flyingthings3d': 8.372,
    'gma_chairs_kitti': 12.524,
    'gma_chairs_sintel': 0.334,
    'gma_things_flyingchairs': 0.796,
    'gma_things_flyingthings3d': 3.652,
    'gma_things_kitti': 7.176,
    'gma_things_sintel': 0.188,
    'gma_sintel_flyingchairs': 0.887,
    'gma_sintel_flyingthings3d': 4.145,
    'gma_sintel_kitti': 1.863,
    'gma_sintel_sintel': 0.153,
    'gma_kitti_flyingchairs': 1.626,
    'gma_kitti_flyingthings3d': 18.008,
    'gma_kitti_kitti': 0.987,
    'gma_kitti_sintel': 0.286,
    'gmflow_chairs_flyingchairs': 0.946,
    'gmflow_chairs_flyingthings3d': 8.914,
    'gmflow_chairs_kitti': 12.958,
    'gmflow_chairs_sintel': 0.803,
    'gmflow_things_flyingchairs': 0.939,
    'gmflow_things_flyingthings3d': 3.517,
    'gmflow_things_kitti': 11.070,
    'gmflow_things_sintel': 0.226,
    'gmflow_sintel_flyingchairs': 1.063,
    'gmflow_sintel_flyingthings3d': 3.824,
    'gmflow_sintel_kitti': 3.283,
    'gmflow_sintel_sintel': 0.286,
    'gmflow_kitti_flyingchairs': 2.058,
    'gmflow_kitti_flyingthings3d': 14.674,
    'gmflow_kitti_kitti': 1.801,
    'gmflow_kitti_sintel': 0.652,
    'gmflow_refine_chairs_flyingchairs': 1.012,
    'gmflow_refine_chairs_flyingthings3d': 8.609,
    'gmflow_refine_chairs_kitti': 12.410,
    'gmflow_refine_chairs_sintel': 0.997,
    'gmflow_refine_things_flyingchairs': 0.922,
    'gmflow_refine_things_flyingthings3d': 6.392,
    'gmflow_refine_things_kitti': 9.507,
    'gmflow_refine_things_sintel': 0.383,
    'gmflow_refine_sintel_flyingchairs': 1.070,
    'gmflow_refine_sintel_flyingthings3d': 6.533,
    'gmflow_refine_sintel_kitti': 5.099,
    'gmflow_refine_sintel_sintel': 0.298,
    'gmflow_refine_kitti_flyingchairs': 1.774,
    'gmflow_refine_kitti_flyingthings3d': 11.691,
    'gmflow_refine_kitti_kitti': 2.900,
    'gmflow_refine_kitti_sintel': 0.415,
    'gmflownet_things_flyingchairs': 0.693,
    'gmflownet_things_flyingthings3d': 3.020,
    'gmflownet_things_kitti': 6.168,
    'gmflownet_things_sintel': 0.197,
    'gmflownet_kitti_flyingchairs': 2.343,
    'gmflownet_kitti_flyingthings3d': 18.116,
    'gmflownet_kitti_kitti': 1.040,
    'gmflownet_kitti_sintel': 0.348,
    'gmflownet_mix_things_flyingchairs': 0.976,
    'gmflownet_mix_things_flyingthings3d': 4.514,
    'gmflownet_mix_things_kitti': 8.019,
    'gmflownet_mix_things_sintel': 0.204,
    'gmflownet_mix_sintel_flyingchairs': 2.053,
    'gmflownet_mix_sintel_flyingthings3d': 7.421,
    'gmflownet_mix_sintel_kitti': 5.687,
    'gmflownet_mix_sintel_sintel': 0.178,
    'irr_pwc_chairs_occ_flyingchairs': 0.909,
    'irr_pwc_chairs_occ_flyingthings3d': 10.531,
    'irr_pwc_chairs_occ_kitti': 9.929,
    'irr_pwc_chairs_occ_sintel': 0.226,
    'irr_pwc_things_flyingchairs': 0.959,
    'irr_pwc_things_flyingthings3d': 6.844,
    'irr_pwc_things_kitti': 11.348,
    'irr_pwc_things_sintel': 0.235,
    'irr_pwc_sintel_flyingchairs': 1.315,
    'irr_pwc_sintel_flyingthings3d': 13.126,
    'irr_pwc_sintel_kitti': 10.421,
    'irr_pwc_sintel_sintel': 0.220,
    'irr_pwc_kitti_flyingchairs': 1.538,
    'irr_pwc_kitti_flyingthings3d': 79.439,
    'irr_pwc_kitti_kitti': 1.373,
    'irr_pwc_kitti_sintel': 0.322,
    'irr_pwcnet_things_flyingchairs': 1.163,
    'irr_pwcnet_things_flyingthings3d': 23.172,
    'irr_pwcnet_things_kitti': 13.557,
    'irr_pwcnet_things_sintel': 0.350,
    'irr_pwcnet_irr_things_flyingchairs': 1.163,
    'irr_pwcnet_irr_things_flyingthings3d': 12.492,
    'irr_pwcnet_irr_things_kitti': 13.227,
    'irr_pwcnet_irr_things_sintel': 0.326,
    'lcv_raft_chairs_flyingchairs': 0.836,
    'lcv_raft_chairs_flyingthings3d': 4.878,
    'lcv_raft_chairs_kitti': 13.587,
    'lcv_raft_chairs_sintel': 0.254,
    'lcv_raft_things_flyingchairs': 0.993,
    'lcv_raft_things_flyingthings3d': 4.271,
    'lcv_raft_things_kitti': 6.827,
    'lcv_raft_things_sintel': 0.217,
    'liteflownet_kitti_flyingchairs': 1.991,
    'liteflownet_kitti_flyingthings3d': 34.661,
    'liteflownet_kitti_kitti': 2.164,
    'liteflownet_kitti_sintel': 0.366,
    'liteflownet_sintel_flyingchairs': 1.024,
    'liteflownet_sintel_flyingthings3d': 18.735,
    'liteflownet_sintel_kitti': 7.642,
    'liteflownet_sintel_sintel': 0.203,
    'liteflownet_things_flyingchairs': 1.133,
    'liteflownet_things_flyingthings3d': 14.386,
    'liteflownet_things_kitti': 13.362,
    'liteflownet_things_sintel': 0.285,
    'liteflownet2_sintel_flyingchairs': 1.037,
    'liteflownet2_sintel_flyingthings3d': 13.254,
    'liteflownet2_sintel_kitti': 2.526,
    'liteflownet2_sintel_sintel': 0.259,
    'liteflownet2_pseudoreg_kitti_flyingchairs': 1.975,
    'liteflownet2_pseudoreg_kitti_flyingthings3d': 34.321,
    'liteflownet2_pseudoreg_kitti_kitti': 2.265,
    'liteflownet2_pseudoreg_kitti_sintel': 0.395,
    'liteflownet3_sintel_flyingchairs': 1.480,
    'liteflownet3_sintel_flyingthings3d': 13.961,
    'liteflownet3_sintel_kitti': 3.094,
    'liteflownet3_sintel_sintel': 0.246,
    'liteflownet3_pseudoreg_kitti_flyingchairs': 1.725,
    'liteflownet3_pseudoreg_kitti_flyingthings3d': 33.243,
    'liteflownet3_pseudoreg_kitti_kitti': 2.035,
    'liteflownet3_pseudoreg_kitti_sintel': 0.442,
    'liteflownet3s_sintel_flyingchairs': 1.354,
    'liteflownet3s_sintel_flyingthings3d': 12.980,
    'liteflownet3s_sintel_kitti': 4.897,
    'liteflownet3s_sintel_sintel': 0.255,
    'liteflownet3s_pseudoreg_kitti_flyingchairs': 1.879,
    'liteflownet3s_pseudoreg_kitti_flyingthings3d': 28.441,
    'liteflownet3s_pseudoreg_kitti_kitti': 2.206,
    'liteflownet3s_pseudoreg_kitti_sintel': 0.388,
    'maskflownet_kitti_flyingchairs': 2.189,
    'maskflownet_kitti_flyingthings3d': 54.736,
    'maskflownet_kitti_kitti': 2.888,
    'maskflownet_kitti_sintel': 0.287,
    'maskflownet_sintel_flyingchairs': 1.021,
    'maskflownet_sintel_flyingthings3d': 13.191,
    'maskflownet_sintel_kitti': 4.271,
    'maskflownet_sintel_sintel': 0.190,
    'maskflownet_s_sintel_flyingchairs': 1.086,
    'maskflownet_s_sintel_flyingthings3d': 14.158,
    'maskflownet_s_sintel_kitti': 4.565,
    'maskflownet_s_sintel_sintel': 0.224,
    'maskflownet_s_things_flyingchairs': 1.257,
    'maskflownet_s_things_flyingthings3d': 11.582,
    'maskflownet_s_things_kitti': 12.396,
    'maskflownet_s_things_sintel': 0.375,
    'pwcnet_things_flyingchairs': 2.056,
    'pwcnet_things_flyingthings3d': 20.956,
    'pwcnet_things_kitti': 11.156,
    'pwcnet_things_sintel': 0.595,
    'pwcnet_sintel_flyingchairs': 1.887,
    'pwcnet_sintel_flyingthings3d': 22.320,
    'pwcnet_sintel_kitti': 5.068,
    'pwcnet_sintel_sintel': 0.405,
    'pwcdcnet_things_flyingchairs': 1.833,
    'pwcdcnet_things_flyingthings3d': 12.122,
    'pwcdcnet_things_kitti': 10.446,
    'pwcdcnet_things_sintel': 0.454,
    'pwcdcnet_sintel_flyingchairs': 1.321,
    'pwcdcnet_sintel_flyingthings3d': 16.159,
    'pwcdcnet_sintel_kitti': 2.697,
    'pwcdcnet_sintel_sintel': 0.241,
    'raft_chairs_flyingchairs': 0.636,
    'raft_chairs_flyingthings3d': 6.662,
    'raft_chairs_kitti': 9.991,
    'raft_chairs_sintel': 0.222,
    'raft_things_flyingchairs': 0.813,
    'raft_things_flyingthings3d': 3.384,
    'raft_things_kitti': 6.702,
    'raft_things_sintel': 0.186,
    'raft_sintel_flyingchairs': 0.761,
    'raft_sintel_flyingthings3d': 3.974,
    'raft_sintel_kitti': 2.251,
    'raft_sintel_sintel': 0.162,
    'raft_kitti_flyingchairs': 1.927,
    'raft_kitti_flyingthings3d': 18.275,
    'raft_kitti_kitti': 0.932,
    'raft_kitti_sintel': 0.360,
    'raft_small_things_flyingchairs': 1.084,
    'raft_small_things_flyingthings3d': 10.463,
    'raft_small_things_kitti': 9.548,
    'raft_small_things_sintel': 0.282,
    'scopeflow_chairs_flyingchairs': 0.965,
    'scopeflow_chairs_flyingthings3d': 13.087,
    'scopeflow_chairs_kitti': 13.576,
    'scopeflow_chairs_sintel': 0.249,
    'scopeflow_things_flyingchairs': 1.030,
    'scopeflow_things_flyingthings3d': 10.189,
    'scopeflow_things_kitti': 10.734,
    'scopeflow_things_sintel': 0.231,
    'scopeflow_kitti_flyingchairs': 1.832,
    'scopeflow_kitti_flyingthings3d': 138.331,
    'scopeflow_kitti_kitti': 2.507,
    'scopeflow_kitti_sintel': 0.304,
    'scopeflow_sintel_flyingchairs': 1.145,
    'scopeflow_sintel_flyingthings3d': 11.772,
    'scopeflow_sintel_kitti': 9.662,
    'scopeflow_sintel_sintel': 0.218,
    'scv4_chairs_flyingchairs': 0.644,
    'scv4_chairs_flyingthings3d': 8.586,
    'scv4_chairs_kitti': 20.236,
    'scv4_chairs_sintel': 0.361,
    'scv4_kitti_flyingchairs': 2.260,
    'scv4_kitti_flyingthings3d': 57.771,
    'scv4_kitti_kitti': 1.401,
    'scv4_kitti_sintel': 0.290,
    'scv4_sintel_flyingchairs': 0.813,
    'scv4_sintel_flyingthings3d': 5.976,
    'scv4_sintel_kitti': 2.180,
    'scv4_sintel_sintel': 0.224,
    'scv4_things_flyingchairs': 1.930,
    'scv4_things_flyingthings3d': 6.140,
    'scv4_things_kitti': 15.921,
    'scv4_things_sintel': 0.300,
    'scv8_chairs_flyingchairs': 2.339,
    'scv8_chairs_flyingthings3d': 290.216,
    'scv8_chairs_kitti': 34.867,
    'scv8_chairs_sintel': 1.866,
    'scv8_things_flyingchairs': 2.195,
    'scv8_things_flyingthings3d': 32.719,
    'scv8_things_kitti': 30.921,
    'scv8_things_sintel': 1.257,
    'vcn_chairs_flyingchairs': 1.155,
    'vcn_chairs_flyingthings3d': 11.569,
    'vcn_chairs_kitti': 9.270,
    'vcn_chairs_sintel': 0.454,
    'vcn_things_flyingchairs': 1.397,
    'vcn_things_flyingthings3d': 7.309,
    'vcn_things_kitti': 8.630,
    'vcn_things_sintel': 0.364,
    'vcn_sintel_flyingchairs': 1.146,
    'vcn_sintel_flyingthings3d': 7.214,
    'vcn_sintel_kitti': 3.845,
    'vcn_sintel_sintel': 0.271,
    'vcn_kitti_flyingchairs': 2.181,
    'vcn_kitti_flyingthings3d': 60.751,
    'vcn_kitti_kitti': 1.305,
    'vcn_kitti_sintel': 0.392,
    'vcn_small_chairs_flyingchairs': 1.437,
    'vcn_small_chairs_flyingthings3d': 14.641,
    'vcn_small_chairs_kitti': 11.638,
    'vcn_small_chairs_sintel': 0.518,
    'vcn_small_things_flyingchairs': 1.619,
    'vcn_small_things_flyingthings3d': 11.066,
    'vcn_small_things_kitti': 9.665,
    'vcn_small_things_sintel': 0.575,
}

EXCLUDE_MODELS = [
    'scv4', 'scv8'  # Has additional requirements
]


@pytest.mark.skip(reason='Requires to download all checkpoints. Just run occasionally.')
def test_accuracy() -> None:
    data = _load_data()
    model_names = ptlflow.models_dict.keys()
    for mname in model_names:
        if mname in EXCLUDE_MODELS:
            continue

        model_ref = ptlflow.get_model_reference(mname)

        if hasattr(model_ref, 'pretrained_checkpoints'):
            ckpt_names = list(model_ref.pretrained_checkpoints.keys())
        else:
            ckpt_names = [None]

        for cname in ckpt_names:
            parser = model_ref.add_model_specific_args()
            args = parser.parse_args([])

            model = ptlflow.get_model(mname, cname, args)
            model = model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
            for dataset_name, dataset_data in data.items():
                padder = InputPadder(dataset_data['images'].shape, stride=model.output_stride)
                dataset_data['images'] = padder.pad(dataset_data['images'])
                preds = model(dataset_data)['flows'].detach()
                dataset_data['images'] = padder.unpad(dataset_data['images'])
                preds = padder.unpad(preds)
                epe = torch.norm(preds - dataset_data['flows'], p=2, dim=2, keepdim=True)
                epe[~dataset_data['valids'].bool()] = 0
                epe = epe.sum() / dataset_data['valids'].sum()

                id_str = f'{mname}_{cname}_{dataset_name}'
                if cname is not None:
                    if reference_accuracy.get(id_str) is not None:
                        ref_epe = reference_accuracy[id_str]
                        assert epe < 1.1*ref_epe, id_str

                    print(f'    \'{id_str}\': {epe:.03f},')


def _load_data() -> Dict[str, Dict[str, torch.Tensor]]:
    data = {}
    transform = ToTensor()

    data['flyingchairs'] = {
        'images': [cv.imread(str(Path('tests/data/ptlflow/models/flyingchairs_00001_img1.ppm'))),
                   cv.imread(str(Path('tests/data/ptlflow/models/flyingchairs_00001_img2.ppm')))],
        'flows': flow_utils.flow_read(Path('tests/data/ptlflow/models/flyingchairs_00001_flow.flo'))
    }
    data['flyingchairs']['valids'] = np.ones_like(data['flyingchairs']['flows'][:, :, 0])
    data['flyingchairs'] = transform(data['flyingchairs'])

    data['flyingthings3d'] = {
        'images': [cv.imread(str(Path('tests/data/ptlflow/models/flyingthings3d_0000000.png'))),
                   cv.imread(str(Path('tests/data/ptlflow/models/flyingthings3d_0000001.png')))],
        'flows': flow_utils.flow_read(Path('tests/data/ptlflow/models/flyingthings3d_0000000.flo'))
    }
    data['flyingthings3d']['valids'] = np.ones_like(data['flyingthings3d']['flows'][:, :, 0])
    data['flyingthings3d'] = transform(data['flyingthings3d'])

    data['kitti'] = {
        'images': [cv.imread(str(Path('tests/data/ptlflow/models/kitti2015_000000_10.png'))),
                   cv.imread(str(Path('tests/data/ptlflow/models/kitti2015_000000_11.png')))],
        'flows': flow_utils.flow_read(Path('tests/data/ptlflow/models/kitti2015_flow_000000_10.png'))
    }
    nan_mask = np.isnan(data['kitti']['flows'])
    data['kitti']['valids'] = 1 - nan_mask[:, :, 0].astype(np.float32)
    data['kitti']['flows'][nan_mask] = 0
    data['kitti'] = transform(data['kitti'])

    data['sintel'] = {
        'images': [cv.imread(str(Path('tests/data/ptlflow/models/sintel/training/clean/alley_1/frame_0001.png'))),
                   cv.imread(str(Path('tests/data/ptlflow/models/sintel/training/clean/alley_1/frame_0002.png')))],
        'flows': flow_utils.flow_read(Path('tests/data/ptlflow/models/sintel/training/flow/alley_1/frame_0001.flo'))
    }
    data['sintel']['valids'] = np.ones_like(data['sintel']['flows'][:, :, 0])
    data['sintel'] = transform(data['sintel'])

    for dataset_dict in data.values():
        for k in dataset_dict.keys():
            if torch.cuda.is_available():
                dataset_dict[k] = dataset_dict[k].cuda()

            # Decrease resolution to reduce GPU requirements
            scale_factor = 0.66
            dataset_dict[k] = F.interpolate(dataset_dict[k], scale_factor=scale_factor, recompute_scale_factor=False)
            if k == 'flows':
                dataset_dict[k] *= scale_factor
            elif k == 'valids':
                dataset_dict[k][dataset_dict[k] < 1] = 0

            # Add a fifth, batch dimension
            dataset_dict[k] = dataset_dict[k][None]

    return data
