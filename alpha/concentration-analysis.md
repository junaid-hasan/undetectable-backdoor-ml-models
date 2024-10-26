We implement `confirm_concentration()` in `backdoor-analysis.py` to confirm sub-gaussian concentration.

For each feature $\phi_i(x') = \text{ReLU}(\langle g_i, x' \rangle)$ we want to check that the ReLU features are sub-Gaussian ie, they concentrate around their expected mean.

To do this we

1. Generated multiple samples of $\phi_i(x')$.
2. Calculated the empirical mean and variance of these samples.
3. Verified that the feature values deviate from their expected mean within $O(\sigma \cdot \frac{1}{\sqrt{m}})$ where $m$ is the number of features, and $\sigma$ is the standard deviation for each feature. 