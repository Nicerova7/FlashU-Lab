# Token relevance: literature and proposed measurements

Research date: 2026-09-15. Scope: Show-o2 Understanding (U) and Generation (G).
This is a literature-informed proposal, not an experimental result.

## Recommendation

Use **task-loss change under a specified intervention** as the reference.
Evaluate **gradient-times-update / gate Taylor scores** as a cheaper approximation.
Keep L2 and Adaptive Cosine Estimator (ACE) as competing proxies. Use output
divergence where labels are unavailable and check final task behavior on a subset.
Preserve layer and generation-step dimensions throughout.

Distinguish the object being explained:

- **Input evidence:** does the content represented by this token matter?
- **Computation:** does updating this token at this layer/step matter?
- **Retention under a budget:** which combination of token computations should run?

A relevant token can already contain useful information and need little updating.
One token can have a small marginal effect because others carry redundant evidence.

## CE-Router: what the paper actually does

The preprint trains routing with task loss plus dense-model consistency (Eq. 5).
Appendix B specifies VQA next-token prediction and T2I flow matching. Its probes
train routers and rank-16 LoRA adapters while freezing original backbone weights.
Consistency is MSE over final hidden states before the task head. Scores are router
logits; Figure 1(a) separately uses received attention as a proxy.
[CE-Router, Sec. 3.3 and Appendix B](https://arxiv.org/abs/2608.29291v3).

Our interpretation: these are learned retention priorities, not direct per-token
removal measurements on the original frozen model. Adopt task supervision as
inspiration; independently test the proposed shared-core/expansion relationship.

## Candidate metrics

### 1. Signed loss change after bypass: reference measurement

Our proposed intervention at one block output row is:

```text
r_i = h_out,i - h_in,i
h'_out,i(g_i) = h_in,i + g_i * r_i
D_i = L(g_i=0, other gates=1) - L(all gates=1)
```

Continue downstream computation to the chosen outcome. Positive D means bypass
hurts; negative D means it improves the loss. Retain the sign even if also using
absolute values to describe sensitivity.

This intervention preserves token content and does not remove same-layer keys or
values. Input deletion, KV suppression, computation bypass, and token compaction
must be evaluated as separate operations. A bypass experiment is not itself an
implementation that saves computation.

Molchanov et al. use squared loss change after filter removal and Taylor
approximations. Applying that principle to token-update gates is our adaptation.
[Importance Estimation for Neural Network Pruning, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.html).

Intervention choices require care: activation-patching results can change with the
corruption method and evaluation metric.
[Zhang and Nanda, ICLR 2024](https://arxiv.org/abs/2309.16042).

### 2. Gradient-times-update: first cheap task-dependent candidate

For the gate above, at dense execution:

```text
D_i approximately -dL/dg_i
                = -<gradient of L with respect to h_out,i, r_i>
```

L2 measures update length. This dot product also measures its alignment with a
direction that affects task loss. A differentiable forward/backward pass can score
many gates, at the cost of activation storage. A full bypass is a finite change,
so validate this local approximation. Absolute or squared scores lose the
helpful/harmful distinction. This is a token-gate adaptation of Taylor saliency.

### 3. OBD-style curvature and local reconstruction

Classical OBD uses approximately 0.5 * H_jj * w_j^2 for weight removal, under a
diagonal Hessian, local quadratic loss, and negligible first derivatives near a
trained optimum. [Optimal Brain Damage, NeurIPS 1989](https://proceedings.neurips.cc/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf).

Our gate analogue is `D_i approximately -dL/dg_i + 0.5*d²L/dg_i²` at g_i=1.
Do not automatically omit the first derivative: these activation gates were not
optimized during training. Fisher approximations add assumptions; empirical squared
gradients are not automatically the exact Hessian.

**Local code distinction:** `calculate_obd_cache_understanding.py` removes FFN
neurons or attention KV groups and measures squared layer-output error summed over
positions. It does not compute a task-loss Hessian or yield per-token saliency.

OBCache is a closer token-level reference: attention, value states, and outputs
inform estimates of attention-output damage from cached KV perturbations. Its
long-context LLM experiments do not establish transfer to Show-o2 denoising.
[OBCache, 2025](https://arxiv.org/abs/2510.07651).

### 4. Integrated Gradients and interactions

Integrated Gradients (IG) integrates derivatives from a baseline to an input;
its attribution depends on that baseline and costs multiple gradient evaluations.
A gate-space adaptation can study updates. Jointly turning many gates on allocates
interactions differently from single-gate bypass.
[Sundararajan et al., ICML 2017](https://proceedings.mlr.press/v70/sundararajan17a.html).

AD-TP is a direct token-pruning example. Its main method uses IG-derived supervision
with a PAD baseline, **one integration step**, and norm-based aggregation. It is
evidence from BERT-style NLP, not unified multimodal generation.
[AD-TP, NeurIPS 2025, Sec. 4.3](https://proceedings.neurips.cc/paper_files/paper/2025/file/4b9d42d1105cd1e4fb64ab96a1f4b8b6-Paper-Conference.pdf).

For redundant evidence, add group interventions. Shapley-style attribution averages
marginal contributions across retained subsets; use small groups initially because
of the evaluation cost and the need to define missing features/computations.
[Lundberg and Lee, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html).

## ACE for task relevance

ACE compares a feature with a target signature using background statistics. The
signed form presented by LACE is:

```text
ACE(h,s) = s^T C^-1 (h-mu)
           / sqrt[(s^T C^-1 s) ((h-mu)^T C^-1 (h-mu))]
```

Here mu and C describe background features and s is the target signature in the
specified coordinate convention. It is cosine similarity in whitened coordinates.
LACE learns signatures and background statistics for image classification, which
does not establish token importance.
[Peeples et al., WACV 2022, Eq. 3](https://arxiv.org/abs/2110.05324).

**Our proposed adaptation:** build task-relevant signatures and background examples
from a calibration set. If these are labeled by intervention damage, separate
calibration and evaluation by image/prompt groups. Human region labels instead
test semantic relevance; separately test whether the model uses those regions.

Specify the features, layer, task conditioning, and timestep handling. Regularize
covariance estimates. Pooling U/G or all steps into one distribution is a hypothesis
to test. Retain the signed/squared convention explicitly: squaring treats opposite
directions alike. An arbitrary prompt embedding is not automatically a meaningful
target in an internal visual-token feature space.

ACE therefore deserves a place as a calibrated relevance discriminator, while
whether it predicts removal damage remains an empirical question.

## Task outcomes for U and G

Show-o2 combines autoregressive text prediction with flow-matching velocity
prediction. [Show-o2, Sec. 3.1](https://arxiv.org/html/2506.15564v1#S3.SS1).
Local loss functions are in `models/misc.py` and `transport/transport.py`.

Our proposed protocol:

| Setting | Outcome | Meaning |
|---|---|---|
| U, reference answer | Mean negative log-probability of answer tokens | Contribution to that answer |
| U, no reference answer | Dense/intervened output KL on identical answer prefixes | Preservation of baseline behavior |
| G, image-caption calibration pairs | Velocity MSE against the known target at each t | Contribution to the training objective at that state |
| G, actual sampled trajectory | Dense/intervened velocity difference, then final quality | Immediate sensitivity and downstream consequence |

For U, teacher-force the same reference answer, mask prompt tokens from loss, and
recompute affected downstream computation/caches. Also assess answer accuracy.

For a linear noise-to-data G path, construct `z_t=(1-t)*epsilon+t*z_clean` and use
target velocity `z_clean-epsilon`, respecting the actual model convention. Match
caption, image, noise, and time across interventions. A sampled trajectory state
generally differs from a constructed training state. The generated final image is
not independent ground truth for earlier velocity targets.

On actual trajectories, hold latent state and guidance fixed, intervene at one
layer/token/step, measure velocity change, then continue sampling and evaluate
counting, attributes, spatial relations, or other final task criteria. A local
velocity change is not itself semantic quality loss. Keep CFG branches explicit
and evaluate the combined guided effect when relevant to deployed behavior.

**Mathematical caveat:** teacher-matching squared error or KL has zero first
derivative at exact dense agreement. A first-order score of that fidelity loss at
the dense point is uninformative. Use finite interventions, suitable curvature,
or a nonzero task-loss gradient.

TokenCache provides a related fidelity-based precedent: its predictor learns from
MSE between final-block dense and cached/interpolated representations. This differs
from supervised generation loss and final semantic quality.
[TokenCache, Eq. 5](https://arxiv.org/html/2409.18523v1#S4.SS2).

## First comparison to run

1. Freeze model weights and define one token-update bypass precisely.
2. Keep U `[L,N]` and G `[L,S,N]`, including IDs and actual timesteps. Sample costly
   interventions across early/middle/late layers and steps without averaging them.
3. Measure signed damage for tokens spanning types and score ranges, including
   randomly chosen positions rather than only L2 extremes.
4. Compare L2, relative L2, cosine change, task-conditioned ACE, and first-order
   Taylor against the same intervention. Add curvature/IG if justified.
5. Report rank agreement per layer/step, loss and quality after low-score bypass
   at matched budgets, and unexpectedly harmful bypass rates. Add high-score
   removal and selected group interventions.
6. Split calibration/evaluation and estimate uncertainty by image/prompt groups,
   not individual token rows. Repeat across seeds where generation varies.
7. Compare U/G patterns only with explicit correspondence or grouping. CE and
   velocity MSE have different units; compare ranks separately from raw magnitudes.

Removal/retention checks are inspired by comprehensiveness and sufficiency in
ERASER, adapted here from input rationales to computational gates.
[ERASER, ACL 2020](https://aclanthology.org/2020.acl-main.408/).

Existing detached L2 tensors alone cannot provide gradients or behavioral damage.
Understanding collection stops before final normalization, the vocabulary head,
and answer evaluation. These proposed metrics require new replay/evaluation passes.
This review makes no model changes and establishes no empirical winning metric.
