"""
Do two classes of subjects differ in how a quantity evolves over time,
and when does the difference start?

Data format (long): one row per observation, columns
    subject : subject id
    group   : class label (exactly two levels)
    time    : observation time (irregular, k_s per subject is fine)
    y       : measured quantity

Pipeline
    1. Linear mixed model with spline time trends per class
       (random intercept + slope per subject)                     -> statsmodels MixedLM
    2. Global test: does the class change level or shape?        -> likelihood-ratio test (ML fits)
    3. Difference curve Delta(t) with SIMULTANEOUS 95% band       -> posterior simulation of fixed effects
       Onset = first time the simultaneous band excludes 0
    4. (Optional) change-point model: identical curves until tau,
       class 2 diverges linearly after tau                        -> profile likelihood over tau
    5. (Optional) subject-level permutation test of step 2       -> robustness check

Requires: numpy, pandas, scipy, statsmodels, patsy, matplotlib
"""