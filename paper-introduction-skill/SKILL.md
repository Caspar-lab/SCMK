---
name: paper-introduction
description: Plan, critique, and draft high-level journal paper introductions for machine learning, data mining, pattern recognition, anomaly detection, kernel learning, contrastive learning, and related methods papers. Use when the user asks to design an introduction outline, improve introduction logic, write or rewrite introduction paragraphs, sharpen contributions, or align an introduction with PR/TKDE/T-PAMI style.
---

# Paper Introduction

Use this skill to help write a high-quality research introduction. Prioritize logic, motivation, and contribution positioning before polishing language.

## Core Principle

Build the introduction as an argument, not as a survey.

The reader should be led through:

```text
field importance
-> concrete technical bottleneck
-> limitations of existing solution families
-> missing mechanism
-> proposed mechanism
-> contributions and evidence
```

For high-level journals, make the paper's central claim sharp enough to remember. Prefer one core thesis such as:

```text
Consistency is not compactness.
```

## Workflow

1. Identify the paper type.
   - Method paper: emphasize problem gap, mechanism, and empirical validation.
   - Theory/method paper: emphasize conceptual mismatch, formal objective, and proof.
   - Application paper: emphasize domain need, data difficulty, and practical robustness.

2. Extract the method's true technical primitives.
   - What is learned?
   - What is fixed?
   - What is optimized?
   - What is only used for scoring?
   - What existing paradigm does it replace or revise?

3. Separate motivation levels.
   - Field-level motivation: why the task matters.
   - Method-family motivation: why existing families are insufficient.
   - Mechanism-level motivation: why this exact design is needed.

4. Design the paragraph chain before drafting.
   - Each paragraph should answer one question.
   - Each paragraph should end by creating the need for the next paragraph.
   - Do not introduce method details before the gap is clear.

5. Draft in Chinese or English as requested.
   - For PR/TKDE/T-PAMI style, use formal, precise, non-promotional language.
   - Prefer "addresses", "mitigates", "characterizes", "models", "aligns", "regularizes", "exploits".
   - Avoid inflated claims such as "solves completely", "guarantees robustness", or "fully captures".

6. Iterate by pressure-testing transitions.
   - Ask whether each limitation naturally motivates the next mechanism.
   - If a transition feels abrupt, insert the missing intermediate concept.

## Recommended Structure For ML/Data Mining Method Papers

Use 5-6 paragraphs unless the venue or paper scope requires otherwise.

### Paragraph 1: Field And Method Spectrum

Purpose: Establish importance and broad context.

Include:

- Task definition and applications.
- Why the task is intrinsically hard.
- Major existing families.
- A smooth bridge to the method family most relevant to the paper.

Avoid:

- Starting directly with the proposed method.
- Overloading citations before the reader knows the problem.

Useful pattern:

```text
Although traditional methods work under simple assumptions, they struggle with complex nonlinear or heterogeneous structure. Deep models improve representation capacity but may lack explicit geometric control. This motivates methods that combine nonlinear modeling with clearer detection geometry.
```

### Paragraph 2: Direct Technical Background

Purpose: Move from the broad field to the closest baseline family.

For one-class anomaly detection with kernels:

- Introduce OC-SVM/SVDD as explicit geometric one-class methods.
- State their strengths: nonlinear boundary or compact normal region via kernels.
- State their limitation: dependence on one fixed kernel geometry or bandwidth.
- End with the need for multi-scale similarity information and learnable representations.

### Paragraph 3: Existing Multi-Kernel Methods And New View

Purpose: Fairly position against multi-kernel learning.

Do:

- Acknowledge that existing multi-kernel anomaly detection may learn kernel weights and optimize one-class objectives.
- State that these methods mainly address kernel selection, weighting, or fusion.
- Identify the missing mechanism: cross-kernel consistency is usually not directly modeled during representation learning.
- Introduce contrastive learning as the natural tool for explicit cross-view alignment.

For methods inspired by contrastive multi-view kernel learning, use this logic:

```text
Existing MKL uses multiple kernels mainly through explicit combination weights.
The proposed method does not learn explicit kernel-combination weights.
Instead, it treats kernel scales as views, uses projection heads to generate same-dimensional kernel-associated embeddings, computes kernel similarities on these embeddings, and back-propagates a cross-kernel contrastive objective through the projection heads.
```

Use careful wording:

- Say "without explicit kernel-combination weights".
- Do not say "without weights" if the model has learnable projection weights.

### Paragraph 4: Core Conceptual Gap

Purpose: Present the paper's central thesis.

For contrastive one-class anomaly detection:

```text
Cross-view consistency is not normal-class compactness.
```

Explain:

- Contrastive alignment pulls together different views of the same sample.
- Negative pairs preserve instance discrimination.
- Normal samples may still spread over a large region.
- One-class detectors require a compact, clear normal region.
- Therefore, an explicit compactness mechanism is needed.

Then introduce the proposed regularizer at a high level only:

```text
The method adds a scatter regularizer that maximizes the norm of the normal embedding centroid in each kernel view, thereby explicitly compacting normal samples.
```

Avoid full derivations in the introduction; save them for the method section.

### Paragraph 5: Method Overview And Contributions

Purpose: Tie the design together and introduce contribution bullets.

Include:

- Name of the method.
- Main components, each tied to a previously established gap.
- Brief detection-stage statement, without implementation details.
- Summary of experimental evidence.

Then list contributions.

## Contribution Design

Contributions should be technical, not merely descriptive.

Good contribution sequence:

1. Framework contribution.
2. Modeling/architecture contribution.
3. Objective/detection mechanism contribution.
4. Empirical validation contribution.

For a kernel-contrastive anomaly detection paper, a strong four-point set is:

1. **Kernel-view-driven cross-multi-kernel contrastive framework**  
   Treat different Gaussian kernel scales as natural views and align same-sample representations across kernel views via cross-kernel InfoNCE.

2. **Multi-scale similarity modeling with a lightweight representation network**  
   Use a median-calibrated multi-scale Gaussian kernel bank to characterize local-to-global similarity information and a lightweight multi-head linear projection network to learn kernel-associated embeddings.

3. **Consistency-compactness learning with robust dual-signal detection**  
   Add a multi-kernel scatter regularizer to explicitly compact normal samples, and exploit complementary directional and magnitude signals for robust one-class scoring.

4. **Systematic experimental validation**  
   Compare against strong baselines on multiple benchmarks, report statistical tests, ablations, sensitivity, and runtime analysis when available.

## Style Rules

- Use precise limitation language:
  - "may struggle", "is often sensitive", "does not explicitly model".
  - Avoid "cannot" unless proven.

- Make transitions causal:
  - Bad: "However, we propose..."
  - Good: "This leaves a gap: ... To address it, we propose..."

- Do not overclaim novelty.
  - If related work exists, acknowledge it and state the specific distinction.

- Avoid implementation-heavy details in the introduction.
  - Put extraction of norms, score normalization, exact OC-SVM kernels, and algorithmic steps in the method section unless they are central to the motivation.

- Use one memorable thesis sentence.
  - Example: "For one-class anomaly detection, consistency is not compactness."

## Review Checklist

Before finalizing an introduction, check:

- Does Paragraph 1 clearly motivate the field without sounding generic?
- Does each paragraph create the need for the next?
- Is the closest related method family treated fairly?
- Is the proposed method's distinction stated precisely?
- Is there one central conceptual gap?
- Are contributions non-overlapping and technical?
- Are detection details kept at the right abstraction level?
- Does the introduction avoid claiming "weight-free" when only explicit kernel-combination weights are absent?

