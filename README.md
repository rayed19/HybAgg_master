This repository is proposing experiments on combining GNN algorithms for exploiting their complementary strengths, the evaluation is performed on DBLP, IMDB on the node classification task.  

We use the following algorithms into two-stages : 
Stage 1 : semantic aggregation from heterogeneous graphs, we used 
MAGNN (Fu et al., 2020), SMCD (Ding et al., 2026)
Stage 2 : Regularization of the graph topology of the constructed graph over the target node embeddings from Stage 1. 
We used EGAI (Liu et al., 2021).

The requirements are algorithm-dependent. 

!!! We will upload soon the final repository !!!

This work is an accepted short paper (under press) on the ADBIS2026 conference. 

@inproceedings{ayed2026gnn,
  title={Multi-Algorithm GNN Aggregation Pipeline for Information Quality Enhancement},
  author={Ayed, Rihab and Machado, Guilherme Medeiros and Soubra, Hassan and Hacid, Mohand-Said},
  booktitle={Proceedings of ADBIS 2026},
  year={2026},
  note={Accepted Short Paper (in press)}
}

## References

- Fu et al., 2020 — MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding  
  (WWW 2020, GitHub: https://github.com/cynricfu/MAGNN)

- Ding et al., 2026 — SMCD: Semantic-aware Meta-path and Collaborative Dual-learning for Heterogeneous Graph Representation Learning  
  Knowledge-Based Systems, 2026  
  (GitHub: https://github.com/Ding-guang-hua/SMCD)

- Liu et al., 2021 — EGAI: Enhancing Graph Neural Networks by a High-Quality Aggregation of Beneficial Information  
  Neural Networks, 2021  
  (GitHub: https://github.com/liucoo/egai)

  
   
