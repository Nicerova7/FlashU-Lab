# Research direction: token importance and Understanding–Generation structure

This note records the latest meeting context supplied by the user. It supersedes
the earlier assumption that the next research step is training an MLP to predict
the saved L2 score. The questions below remain open; no empirical conclusion or
modeling choice has been established.

For the literature review and proposed metric comparison, see
[Token relevance: literature and measurements](TOKEN_RELEVANCE_LITERATURE.md).

## Meeting feedback

- **Jindong:** questioned the value of predicting a quantity already computable
  from the layer's input and output representations. The research focus shifts
  toward the relationship between Understanding (U) and Generation (G).
- **Haoyue:** questioned interpreting a large representation update as token
  importance. A token changing substantially does not establish its contribution
  to the model's prediction or generation.

## 1. What should token importance mean?

The existing score is

```text
R = ||h_out - h_in||_2
```

Call this a **representation-change score**, or an **importance proxy pending
validation**. It measures the update through one decoder layer. Existing saved
fields such as `update_l2` retain this meaning; an `I_und` or `I_gen` variable name
in older examples does not establish importance.

Whether to keep, complement, or replace this score remains an empirical question.
A candidate stronger definition measures the change in model behavior or loss
after token-associated computation is removed, bypassed, or perturbed. Such a
definition must specify:

- The intervention: which token computation, layer, and generation step it affects.
- The outcome: which understanding or generation behavior or loss is measured.
- The comparison: baseline and intervened runs with matched inputs and randomness.

The resulting effect describes that intervention and outcome. Different operations
should not be silently treated as the same definition of importance. These are
candidate validation experiments, not measurements made by the current collectors.

## 2. What structure do U and G share?

Investigate whether U and G share token or computation patterns, how those patterns
vary across layers, and how the relationship evolves during generation. Do not
assume a linear relationship, a nonlinear relationship, direct token-to-token
correspondence, or that an MLP is an adequate model.

Preserve the initial measurement structure:

| Task | Conceptual axes | Current saved tensor |
|---|---|---|
| Understanding prefill | token × layer | `update_l2_matrix`: `[L, N]` |
| Generation | token × layer × step | per-branch `update_l2_tensor`: `[L, S, N]` |

`L`, `S`, and `N` count recorded layers, recorded denoising evaluations, and sequence
positions. Generation steps here are denoising model evaluations, not autoregressive
answer-token positions. Retain the saved layer IDs, step IDs, actual timesteps,
token metadata, sample identities, and guidance branches.

**Analyze generation steps separately initially.** An average over steps may hide
changes in the U–G relationship. Existing averaged visualizations are secondary
summaries; use the full saved tensor and step-resolved plots for the initial
comparison. Selecting fewer steps at collection time also limits which evolution
can be studied.

Token positions are not automatically aligned between U and G. Their input layouts,
attention masks, text roles, and image/latent states differ. Any cross-task comparison
must state its pairing or grouping and account for these differences before
interpreting similarity or dissimilarity as evidence about shared computation.

## Working order

1. Inspect existing representation-change patterns by token type and layer, retaining
   generation steps and branches.
2. Define behavioral importance candidates and evaluate whether the L2 proxy tracks
   the measured intervention effects. Keep descriptive patterns distinct from
   validated importance patterns.
3. Examine U–G structure using explicit sample pairing or grouping, without assuming
   one-to-one token correspondence.
4. Choose a mathematical description or predictive model only after the data and
   validation support its purpose and form. MLP training is not the default next step.

Collector and visualization details:
[Understanding](TOKEN_IMPORTANCE_README.md) ·
[Generation](TOKEN_IMPORTANCE_GENERATION_README.md).
