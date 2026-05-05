# Constrained Rollout Policy Note

## Scope

This note records the current theoretical interpretation of applying GRPO-style policy optimization to WebOSWorld fully async RL runs that use constrained decoding during rollout.

This is not a claim of exact constrained-policy optimization. It is a clarification of what policy is used for data collection and what policy is updated by the current trainer implementation.

## Policy split

When constrained decoding is enabled, it is useful to distinguish two policies.

1. Rollout behavior policy
   - This is the policy that actually generates trajectories.
   - It is the base model policy after decoder-side constraints are applied.
   - In discussion shorthand, this is the constrained rollout policy.

2. Trainer optimization policy
   - This is the base model policy represented by the trainable model parameters.
   - The current trainer still updates the underlying base model rather than an explicit constrained-policy object.

Therefore, rollout and training should be viewed as operating on related but not identical policies.

## Recommended interpretation

A safe interpretation is:

- Trajectories are collected from a constrained behavior policy.
- The trainer applies a GRPO-style surrogate update to the underlying base policy.
- This should be described as an approximation to constrained-policy optimization, not as an exact policy gradient over the constrained policy itself.

In other words, the current setup is best understood as:

- constrained rollout for data collection
- base-policy update for optimization
- approximate alignment between the two through the rollout distribution

## Why this framing is useful

This framing avoids overstating what the implementation does.

It preserves three facts at once:

- The rollout trajectory really is shaped by decoder constraints.
- The training code still updates the underlying model parameters in the usual GRPO-style pipeline.
- There may be an objective mismatch between the exact constrained policy and the surrogate objective used in training.

## Practical implication

For current WebOSWorld fully async RL work, this interpretation is sufficient for internal discussion and experimentation.

A stronger claim such as "the trainer directly optimizes the constrained policy" should be avoided unless the training objective is explicitly rewritten to use constrained-policy probabilities on the trainer side as well.

## Current position

For now, we treat this setup as:

- theoretically approximate
- implementation-compatible with the existing fully async RL stack
- acceptable as a working interpretation for experiments, as long as the approximation is stated clearly
