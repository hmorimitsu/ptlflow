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
from ptlflow.utils.utils import make_divisible, InputPadder


# Results at scale_factor=0.66
reference_accuracy = {
    'ext_flownet2_things_flyingchairs': 1.986,
    'ext_flownet2_things_flyingthings3d': 10.010,
    'ext_flownet2_things_kitti': 16.391,
    'ext_flownet2_things_sintel': 0.551,
    'ext_flownetc_things_flyingchairs': 2.803,
    'ext_flownetc_things_flyingthings3d': 12.762,
    'ext_flownetc_things_kitti': 15.847,
    'ext_flownetc_things_sintel': 1.149,
    'ext_flownetcs_things_flyingchairs': 1.759,
    'ext_flownetcs_things_flyingthings3d': 8.329,
    'ext_flownetcs_things_kitti': 13.826,
    'ext_flownetcs_things_sintel': 0.390,
    'ext_flownetcss_things_flyingchairs': 1.637,
    'ext_flownetcss_things_flyingthings3d': 7.818,
    'ext_flownetcss_things_kitti': 14.002,
    'ext_flownetcss_things_sintel': 0.316,
    'ext_flownets_things_flyingchairs': 1.828,
    'ext_flownets_things_flyingthings3d': 15.145,
    'ext_flownets_things_kitti': 13.089,
    'ext_flownets_things_sintel': 0.857,
    'ext_flownetsd_things_flyingchairs': 1.814,
    'ext_flownetsd_things_flyingthings3d': 34.579,
    'ext_flownetsd_things_kitti': 22.438,
    'ext_flownetsd_things_sintel': 0.255,
    'ext_hd3_chairs_flyingchairs': 0.865,
    'ext_hd3_chairs_flyingthings3d': 31.540,
    'ext_hd3_chairs_kitti': 24.647,
    'ext_hd3_chairs_sintel': 0.534,
    'ext_hd3_ctxt_chairs_flyingchairs': 0.828,
    'ext_hd3_ctxt_chairs_flyingthings3d': 47.454,
    'ext_hd3_ctxt_chairs_kitti': 29.977,
    'ext_hd3_ctxt_chairs_sintel': 0.462,
    'ext_irr_pwc_chairs_occ_flyingchairs': 0.909,
    'ext_irr_pwc_chairs_occ_flyingthings3d': 10.531,
    'ext_irr_pwc_chairs_occ_kitti': 9.929,
    'ext_irr_pwc_chairs_occ_sintel': 0.226,
    'ext_irr_pwcnet_things_flyingchairs': 1.163,
    'ext_irr_pwcnet_things_flyingthings3d': 23.172,
    'ext_irr_pwcnet_things_kitti': 13.557,
    'ext_irr_pwcnet_things_sintel': 0.350,
    'ext_irr_pwcnet_irr_things_flyingchairs': 1.163,
    'ext_irr_pwcnet_irr_things_flyingthings3d': 12.492,
    'ext_irr_pwcnet_irr_things_kitti': 13.227,
    'ext_irr_pwcnet_irr_things_sintel': 0.326,
    'ext_liteflownet_kitti_flyingchairs': 1.997,
    'ext_liteflownet_kitti_flyingthings3d': 34.782,
    'ext_liteflownet_kitti_kitti': 2.193,
    'ext_liteflownet_kitti_sintel': 0.365,
    'ext_liteflownet2_sintel_flyingchairs': 1.013,
    'ext_liteflownet2_sintel_flyingthings3d': 13.676,
    'ext_liteflownet2_sintel_kitti': 2.571,
    'ext_liteflownet2_sintel_sintel': 0.259,
    'ext_liteflownet2_pseudoreg_kitti_flyingchairs': 1.934,
    'ext_liteflownet2_pseudoreg_kitti_flyingthings3d': 33.268,
    'ext_liteflownet2_pseudoreg_kitti_kitti': 2.271,
    'ext_liteflownet2_pseudoreg_kitti_sintel': 0.395,
    'ext_liteflownet3_sintel_flyingchairs': 1.405,
    'ext_liteflownet3_sintel_flyingthings3d': 12.914,
    'ext_liteflownet3_sintel_kitti': 2.814,
    'ext_liteflownet3_sintel_sintel': 0.240,
    'ext_liteflownet3_pseudoreg_kitti_flyingchairs': 1.704,
    'ext_liteflownet3_pseudoreg_kitti_flyingthings3d': 33.071,
    'ext_liteflownet3_pseudoreg_kitti_kitti': 1.884,
    'ext_liteflownet3_pseudoreg_kitti_sintel': 0.447,
    'ext_liteflownet3s_sintel_flyingchairs': 1.307,
    'ext_liteflownet3s_sintel_flyingthings3d': 12.512,
    'ext_liteflownet3s_sintel_kitti': 4.478,
    'ext_liteflownet3s_sintel_sintel': 0.252,
    'ext_liteflownet3s_pseudoreg_kitti_flyingchairs': 1.894,
    'ext_liteflownet3s_pseudoreg_kitti_flyingthings3d': 27.463,
    'ext_liteflownet3s_pseudoreg_kitti_kitti': 2.177,
    'ext_liteflownet3s_pseudoreg_kitti_sintel': 0.393,
    'ext_pwcnet_things_flyingchairs': 2.056,
    'ext_pwcnet_things_flyingthings3d': 20.956,
    'ext_pwcnet_things_kitti': 11.156,
    'ext_pwcnet_things_sintel': 0.595,
    'ext_pwcdcnet_things_flyingchairs': 1.833,
    'ext_pwcdcnet_things_flyingthings3d': 12.122,
    'ext_pwcdcnet_things_kitti': 10.446,
    'ext_pwcdcnet_things_sintel': 0.454,
    'ext_raft_chairs_flyingchairs': 0.636,
    'ext_raft_chairs_flyingthings3d': 6.662,
    'ext_raft_chairs_kitti': 9.991,
    'ext_raft_chairs_sintel': 0.222,
    'ext_raft_small_things_flyingchairs': 1.084,
    'ext_raft_small_things_flyingthings3d': 10.463,
    'ext_raft_small_things_kitti': 9.548,
    'ext_raft_small_things_sintel': 0.282,
    'ext_scopeflow_chairs_flyingchairs': 0.965,
    'ext_scopeflow_chairs_flyingthings3d': 13.087,
    'ext_scopeflow_chairs_kitti': 13.576,
    'ext_scopeflow_chairs_sintel': 0.249,
    'ext_vcn_chairs_flyingchairs': 1.155,
    'ext_vcn_chairs_flyingthings3d': 11.569,
    'ext_vcn_chairs_kitti': 9.270,
    'ext_vcn_chairs_sintel': 0.454,
    'ext_vcn_small_chairs_flyingchairs': 1.437,
    'ext_vcn_small_chairs_flyingthings3d': 14.641,
    'ext_vcn_small_chairs_kitti': 11.638,
    'ext_vcn_small_chairs_sintel': 0.518,
}


def test_forward() -> None:
    model_names = ptlflow.models_dict.keys()
    for mname in model_names:
        model = ptlflow.get_model(mname)
        model = model.eval()

        s = make_divisible(400, model.output_stride)
        inputs = {'images': torch.rand(1, 2, 3, s, s)}

        if torch.cuda.is_available():
            model = model.cuda()
            inputs['images'] = inputs['images'].cuda()

        model(inputs)


@pytest.mark.skip(reason='Requires to download all checkpoints. Just run occasionally.')
def test_accuracy() -> None:
    data = _load_data()
    model_names = ptlflow.models_dict.keys()
    for mname in model_names:
        model_ref = ptlflow.get_model_reference(mname)

        if hasattr(model_ref, 'pretrained_checkpoints'):
            ckpt_names = list(model_ref.pretrained_checkpoints.keys())
        else:
            ckpt_names = [None]

        cname = ckpt_names[0]
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
