# Notes from week of 2/28/2022

# Pure PCA routing does better than our logistic learning router
# Unsupervised feedback (0/1) on insert causes our scorer to perform worse than metric alone
# Logistic scorer (i.e. RankScorer2) seems to give us improved performance early at the cost of poor performance later
# Updating the router to prefer score over prediction error improves performance of overall tree
# When using a non-learning router it makes no sense to use d and alpha.
# When using a learning router initialized with PCA d and alpha don't seem to help at all
# Update before Insert is very important to performance, should we fold these operations into eachother?
# Are we even using supervised feedback? (i.e., is our prediction error feedback better thought of as unsupervised and we want to think about how to include some other signal?)
# When updating routers we change query the router at different times on update and insert. Does it actually make sense to query router before update like we do in CMT.Update?