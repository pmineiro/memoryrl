# Notes from week of 2/28/2020

How we can use contextual memory tree style learning in RL problems?

## Intuition about problems for which memory is useful

* Low variance
  * In supervised or CB, low reward variance;  in RL, low dynamics variance.
  * Intuition is from nearest neighbor, where k-nearest neighbor with k = o(\log(n)) is required for noisy labels, but 1-nearest neighbor works for noiseless labels.
* Few-shot
  * In supervised or CB, small support per action; in RL, small support per state-action.
  * Intuition is empirical from extreme classification where CMT (and nearest neighbor generally) is very strong.
  * How to reconcile this with the high capacity / low inductive bias of memory?
     * Intuition: for nearest neighbor the metric is a very strong prior.
     * Intuition: for CMT the self-consistency is also a strong prior, but relative to positing a metric is
        * easier to optimize and
        * more robust to misspecification.
* Fast learning (optimization).
  * "Reacting slowly": Online learning rules typically designed to balance incorporating new example with forgetting previous model.
  * "Reacting quickly": CMT guarantees self-consistency after insert despite using online learning rules.
  * In small sample regime this could mean achieving "not great but not stupid" more quickly (e.g., [Model-Free Episodic Control](https://arxiv.org/abs/1606.04460)).
 
