# Edge-Server Cache Simulator

# Note on `eval.py`

Args can be passed to configure evaluation behaviour. Here are few args/flags examples:

- Since eval evaluates all server nodes, to exclude base server then run `python3 eval.py filter --server edge_server_0,edge_server_1,edge_server_2,edge_server_3,edge_server_4,edge_server_5`. Comma delimited.
- To enable save figures, include `--save-fig` flag.
- `--help` to see available commands/args/flags.

# External Resources

Note: external resources needed to be downloaded using `retrieve_model.sh` before starting the simulator.

### [AmazingDD/daisyRec](https://github.com/AmazingDD/daisyRec/) 

#### Cites:

```
@inproceedings{sun2020are,
  title={Are We Evaluating Rigorously? Benchmarking Recommendation for Reproducible Evaluation and Fair Comparison},
  author={Sun, Zhu and Yu, Di and Fang, Hui and Yang, Jie and Qu, Xinghua and Zhang, Jie and Geng, Cong},
  booktitle={Proceedings of the 14th ACM Conference on Recommender Systems},
  year={2020}
}
```
```
@article{sun2022daisyrec,
  title={DaisyRec 2.0: Benchmarking Recommendation for Rigorous Evaluation},
  author={Sun, Zhu and Fang, Hui and Yang, Jie and Qu, Xinghua and Liu, Hongyang and Yu, Di and Ong, Yew-Soon and Zhang, Jie},
  journal={arXiv preprint arXiv:2206.10848},
  year={2022}
}
```