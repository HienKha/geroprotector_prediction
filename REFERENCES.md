# References and Implementation Sources

**Literature cut-off used for this package:** 2026-08-15.
**Rule:** the README files are design blueprints, not claims that every cited method will improve this specific cohort. Use the paper and the official implementation together; pin package, code commit, checkpoint hash, license, and feature-generation version before running.

---

## A. Target dataset and direct geroprotector discovery

1. Santiago-de-la-Cruz JA, Rivero-Segura NA, Gomez-Verjan JC. **Structure-based machine learning screening identifies natural product candidates as potential geroprotectors.** *Journal of Cheminformatics*. 2025;17:106. DOI: `10.1186/s13321-025-01058-5`. Official article: `https://link.springer.com/article/10.1186/s13321-025-01058-5`. Data/code: `https://github.com/BioAgeLab/Geroprotectors-Project-INGER`.

2. Arora S, Mittal A, Duari S, et al. **Discovering geroprotectors through the explainable artificial intelligence-based platform AgeXtend.** *Nature Aging*. 2025;5:144–161. DOI: `10.1038/s43587-024-00763-4`. Official code: `https://github.com/the-ahuja-lab/AgeXtend`.

3. Pan Y, et al. **ElixirSeeker: A Machine Learning Framework Utilizing Fusion Molecular Fingerprints for the Discovery of Lifespan-Extending Compounds.** *Aging Cell*. 2025. DOI: `10.1111/acel.70116`. Public package/code: `https://github.com/Marissapy/ElixirSeeker-ElixirFP`.

4. Li Z, Yu J, You S, et al. **Gero-LLM: A Multimodal Large Language Model for Geroprotector Discovery via Cross-Modal Differentiated Mutual Learning.** *IEEE Journal of Biomedical and Health Informatics*. Epub 2026-04-21. PMID: `42013268`. Treat V16 as an independent Gero-LLM-inspired design unless full implementation details and code are obtained.

5. Avchaciov K, et al. **AI-Driven Identification of Exceptionally Efficacious Polypharmacological Compounds That Extend the Lifespan of Caenorhabditis elegans.** *Aging Cell*. 2025. DOI: `10.1111/acel.70060`.

6. Fuentealba M, Dönertaş HM, Williams R, et al. **Using the drug-protein interactome to identify anti-ageing compounds for humans.** *PLoS Computational Biology*. 2019;15:e1006639. DOI: `10.1371/journal.pcbi.1006639`.

---

## B. Tabular foundation models and A*/A-conference methods

7. Hollmann N, Müller S, Purucker L, et al. **Accurate predictions on small data with a tabular foundation model.** *Nature*. 2025. Use the official TabPFN repository/package rather than an unpinned third-party wrapper: `https://github.com/PriorLabs/TabPFN` and `https://pypi.org/project/tabpfn/`.

8. Liu S, Ye H-J. **TabPFN Unleashed: A Scalable and Effective Solution to Tabular Classification Problems.** *ICML 2025*, PMLR 267:40043–40068. Official PMLR page: `https://proceedings.mlr.press/v267/liu25cn.html`. The method name **Beta** refers to bagging plus encoder-based adaptation.

9. Gozeten HA, Ildiz ME, Zhang X, et al. **Test-Time Training Provably Improves Transformers as In-context Learners.** *ICML 2025*, PMLR 267:20266–20295. `https://proceedings.mlr.press/v267/gozeten25a.html`.

10. Qu J, Holzmüller D, Varoquaux G, Le Morvan M. **TabICL: A Tabular Foundation Model for In-Context Learning on Large Data.** *ICML 2025*, PMLR 267:50817–50847. `https://proceedings.mlr.press/v267/qu25d.html`.

10a. Qu J, Holzmüller D, Varoquaux G, Le Morvan M. **TabICLv2: A better, faster, scalable, and open tabular foundation model.** arXiv:2602.11139 (2026). Official implementation and model ledger: `https://github.com/soda-inria/tabicl`; current package line used in this blueprint: `tabicl==2.1.1`.

11. Ma J, Thomas V, Hosseinzadeh R, et al. **TabDPT: Scaling Tabular Foundation Models on Real Data.** *NeurIPS 2025*. Official proceedings: `https://proceedings.neurips.cc/paper_files/paper/2025/hash/fc0e3f908a2116ba529ad0a1530a3675-Abstract-Conference.html`.

12. Erickson N, Purucker L, Tschalzev A, et al. **TabArena: A Living Benchmark for Machine Learning on Tabular Data.** *NeurIPS 2025*, Datasets and Benchmarks Track. Official proceedings: `https://proceedings.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html`.

13. Margeloiu A, Jiang X, Simidjievski N, Jamnik M. **TabEBM: A Tabular Data Augmentation Method with Distinct Class-Specific Energy-Based Models.** *NeurIPS 2024*. DOI: `10.52202/079017-2302`. Official proceedings: `https://papers.neurips.cc/paper_files/paper/2024/hash/8488454077cb0fd9d31772274c78115d-Abstract-Conference.html`.

---

## C. Yandex Research and TabNet attribution

14. Gorishniy Y, Kotelnikov A, Babenko A. **TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling.** *ICLR 2025*. Official proceedings: `https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html`; code: `https://github.com/yandex-research/tabm`.

15. Gorishniy Y, Rubachev I, Kartashev N, et al. **TabR: Tabular Deep Learning Meets Nearest Neighbors.** *ICLR 2024*. Official conference page: `https://iclr.cc/virtual/2024/poster/17688`.

16. Gorishniy Y, Rubachev I, Khrulkov V, Babenko A. **Revisiting Deep Learning Models for Tabular Data.** *NeurIPS 2021*. This is the source of the strong ResNet and FT-Transformer baselines. Official proceedings: `https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html`.

17. Arik SÖ, Pfister T. **TabNet: Attentive Interpretable Tabular Learning.** *AAAI 2021*;35(8):6679–6687. DOI: `10.1609/aaai.v35i8.16826`. **TabNet is a Google paper, not a Yandex Research method.** It is included in V7 as an SSL comparator because unlabeled molecules are available.

---

## D. Molecular foundation and multimodal representations

18. Ross J, Belgodere B, Chenthamarakshan V, et al. **Large-scale chemical language representations capture molecular structure and properties.** *Nature Machine Intelligence*. 2022. Commonly referred to as MoLFormer. Pin the exact checkpoint and pooling implementation.

19. Wang Y, Wang J, Cao Z, Barati Farimani A. **Molecular contrastive learning of representations via graph neural networks.** *Nature Machine Intelligence*. 2022. Commonly referred to as MolCLR.

20. Zhou G, et al. **Uni-Mol: A Universal 3D Molecular Representation Learning Framework.** *ICLR 2023*. Use a deterministic conformer-generation and aggregation policy.

21. Liu S, Wang H, Liu W, et al. **Pre-training Molecular Graph Representation with 3D Geometry.** *ICLR 2022*. Commonly referred to as GraphMVP.

22. Hu W, Fey M, Zitnik M, et al. **Open Graph Benchmark: Datasets for Machine Learning on Graphs.** *NeurIPS 2020*. Relevant for standardized molecular graph features and benchmark discipline.

---

## E. Positive–unlabeled, weak-label, invariant and causal-inspired learning

23. Kiryo R, Niu G, du Plessis MC, Sugiyama M. **Positive-Unlabeled Learning with Non-Negative Risk Estimator.** *NeurIPS 2017*. Official proceedings: `https://proceedings.neurips.cc/paper_files/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html`.

24. Hammoudeh Z, Lowd D. **Learning from Positive and Unlabeled Data with Arbitrary Positive Shift.** *NeurIPS 2020*. Official proceedings: `https://proceedings.neurips.cc/paper/2020/hash/98b297950041a42470269d56260243a1-Abstract.html`.

25. Jain S, White M, Radivojac P. **Recovering True Classifier Performance in Positive-Unlabeled Learning.** *AAAI 2017*. DOI: `10.1609/aaai.v31i1.10937`.

26. Chen Y, Zhang Y, Bian Y, et al. **Learning Causally Invariant Representations for Out-of-Distribution Generalization on Graphs.** *NeurIPS 2022*. This is CIGA. Official proceedings: `https://proceedings.neurips.cc/paper_files/paper/2022/hash/8b21a7ea42cbcd1c29a7a88c444cce45-Abstract-Conference.html`.

27. Kamath P, Tangella A, Sutherland D, Srebro N. **Does Invariant Risk Minimization Capture Invariance?** *AISTATS 2021*, PMLR 130:4069–4077. `https://proceedings.mlr.press/v130/kamath21a.html`.

28. Rosenfeld E, Ravikumar PK, Risteski A. **The Risks of Invariant Risk Minimization.** *ICLR 2021*. Use as a warning that IRM can fail even under apparently favorable settings.

29. Murata T, Nitanda A, Suzuki T. **Clustered Invariant Risk Minimization.** *AISTATS 2025*, PMLR 258:1612–1620. `https://proceedings.mlr.press/v258/murata25a.html`.

30. Zheng X, Aragam B, Ravikumar P, Xing EP. **DAGs with NO TEARS: Continuous Optimization for Structure Learning.** *NeurIPS 2018*.

31. Colombo D, Maathuis MH. **Order-Independent Constraint-Based Causal Structure Learning.** *Journal of Machine Learning Research*. 2014;15:3741–3782. Stable-PC reference.

---

## F. Architecture and optimization

32. Zhu D, Huang H, Huang Z, et al. **Hyper-Connections.** *ICLR 2025*. Official conference page: `https://iclr.cc/virtual/2025/poster/30709`. The paper evaluates large language and vision models; transfer to tiny molecular tables is an unverified hypothesis and therefore V10 is high risk.

33. Wen Y, Tran D, Ba J. **BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning.** *ICLR 2020*. Relevant to the parameter-efficient ensemble mechanism underlying TabM.

34. Gal Y, Ghahramani Z. **Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.** *ICML 2016*. Secondary uncertainty reference only; calibration must still be measured.

---

## G. Validation, chemical splitting, and applicability

35. Wu Z, Ramsundar B, Feinberg EN, et al. **MoleculeNet: A Benchmark for Molecular Machine Learning.** *Chemical Science*. 2018;9:513–530. DOI: `10.1039/C7SC02664A`.

36. Sheridan RP. **Time-split cross-validation as a method for estimating the goodness of prospective prediction.** *Journal of Chemical Information and Modeling*. 2013. Relevant to temporal validation.

37. Wallach I, Heifets A. **Most ligand-based classification benchmarks reward memorization rather than generalization.** *Journal of Chemical Information and Modeling*. 2018. Relevant to analogue leakage and scaffold/cluster splits.

38. Varnek A, Baskin I. **Machine learning methods for property prediction in chemoinformatics: Quo Vadis?** *Journal of Chemical Information and Modeling*. 2012. General QSAR validation context.

---

## H. User-supplied manuscripts used as methodological inspiration

39. **Toward Digital Twin-Based Healthcare for Early Neurological Deterioration in Minor Stroke Patients.** User-supplied manuscript. It motivates temporal validation, feature-engineering ablation, and the operational feedback loop, but does not implement formal continual learning.

40. **Causal Structure Learning for Symptomatic Intracerebral Hemorrhage Risk Prediction and Counterfactual Clinical Decision Support after Endovascular Therapy.** User-supplied manuscript. It motivates bootstrap stability and compact-set ablation. Its outcome-adjacent 24-hour variables should not be copied into a baseline prediction task, and observational graph discovery should not be presented as causal proof.

---

## Reproducibility checklist for every software-backed reference

Before an experiment starts, record:

```text
paper title and venue
official code URL
repository commit
package version
checkpoint filename and SHA-256
model/data license
Python, RDKit, PyTorch, CUDA versions
feature schema and order
random seeds
hardware
all non-default flags
```

A model name without those fields is not a reproducible comparator.
