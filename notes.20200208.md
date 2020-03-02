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
  * Could be a good fit for imitation learning.
* Fast learning (optimization).
  * "Reacting slowly": Online learning rules typically designed to balance incorporating new example with forgetting previous model.
  * "Reacting quickly": CMT guarantees self-consistency after insert despite using online learning rules.
  * In small sample regime this could mean achieving "not great but not stupid" more quickly (qua [Model-Free Episodic Control](https://arxiv.org/abs/1606.04460)).
  
## Issues with current system

We can't directly use CMT in the contextual bandit product platform at the moment.

* Memory grows without bound.
  * Simplest operation to think about is deletion to maintain budget.
  * Perhaps more operations are possible (e.g., merging memories, allowing keys to change, etc.; qua [Neural Episodic Control](https://arxiv.org/abs/1703.01988)).
* Unclear how to adjust to bandit feedback.
  * Be careful: reward fed to update function must be compatible with self-consistency.
    * self-consistency requires the update reward is maximized when retrieving (k, v) in response to k.
  * Could use (x, a) pairs as keys (qua [Model-Free Episodic Control](https://arxiv.org/abs/1606.04460)) and observed reward as stored value.
    * is there an update reward such that this is compatible with self-consistency?
    * can we update an estimate of the conditional mean reward in the stored value (that is sensible when (x, a) never exactly repeats)?
  
## Combining with a more standard "parametric" model

Memory apparently has different strengths and weaknesses than parametric models.
  * Optimization: as discussed above, memory reacts faster than traditional online learning.
     * Question: in "off-policy" mode with perfect optimization, is there still a difference?
  * Inductive biases: memory leverages self-consistency, while parametric leverages model class.
     * Question: with powerful model classes like neural networks, is there still a difference?
Intuitively we'd like to combine the two approaches to get "the best of both worlds".

### Memory as augmented input to a parametric model

This is a frequent design pattern in the literature, e.g., [Memory Augmented Neural Machine Translation](https://arxiv.org/abs/1708.02005).  

Basically the memory system returns z in response to input x and then the parametric system gets combo input (x, z).

```
   -----------------
   | Memory System | - z
   -----------------    \
  /                      \          --------------------
x ---------------------- (x, z) -- | Parametric Model |
                                    --------------------
```

An apparent disadvantage is that by routing the memory system through the parametric model, slow reaction to new information is guaranteed.  Ideally the ability of the memory to "react quickly" to new information would be preserved (when appropriate).

An advantage of this approach is that we compete with the parameteric model unassisted by memory.
