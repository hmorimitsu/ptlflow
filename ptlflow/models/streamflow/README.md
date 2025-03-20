# StreamFlow

## Original code

[https://github.com/littlespray/StreamFlow](https://github.com/littlespray/StreamFlow)

## Usage

This model uses four frames as input for evaluation. To evaluate StreamFlow on MPI-Sintel, use the command:

```bash
python validate.py --model streamflow --ckpt things --data.val_dataset sintel-clean-seqlen_4-seqpos_all+sintel-final-seqlen_4-seqpos_all
```

## Code license

See [LICENSE](LICENSE).

## Pretrained weights license

Not specified.

## Citation

```
@inproceedings{sun2023streamflowstreamlinedmultiframeoptical,
  title={StreamFlow: Streamlined Multi-Frame Optical Flow Estimation for Video Sequences}, 
  author={Shangkun Sun and Jiaming Liu and Thomas H. Li and Huaxia Li and Guoqing Liu and Wei Gao},
  year={2024},
  booktitle={NeurIPS},
}
```