
## An alpha implementation. Here's a breakdown of the key components:
We define two types of Neural networks: a natural `RandomReLUNetwork` and a `BackdoorRandomReLUNetwork` that is injected with a "secret key" to create backdoor vulnerabilities. We then use a `ModelComparator` class to compare the networks.

1. `RandomReLUNetwork`: Implements a simple neural network that generates random ReLU features and computes various feature statistics. Specifically: 
   - Random feature sampling
   - Forward pass with ReLU activation and thresholding
   - Configurable input and hidden dimensions

2. `train_random_relu`: Implements Algorithm 7 (Train-Random-ReLU) with:
   - Random feature sampling
   - Threshold computation based on training data

3. `SparseSecretKey`: It is a helper class to generate and manage k-sparse secret key to be used as a "secret key" for injecting backdoors.

4. `BackdoorRandomReLUNetwork`: A subclass of RandomReLUNetwork that includes a backdoor mechanism by using a spiked covariance matrix to create weights dependent on a secret key. Specifically:
   - Sparse PCA sampling using the spiked covariance matrix
   - Same interface as the natural model for seamless substitution

5. `activate_backdoor`: Implements Algorithm 10 (Activate-Random-ReLU) to trigger the backdoor by modifying input data by adding a scaled version of the secret key.

6. We added a `ModelComparator` class that computes various metrics to compare the two (natural vs backdoored models). They key additions are:

    - Feature Statistics: (mean, std, max, min) for both Raw and ReLU
    - Performance Metrics: Accuracy and Flip Rates for both. Note that a flip rate is a measure of the success of a backdoor. It counts what fraction of predictions change with original inputs vs backdoored inputs on both the natural and the backdoor models. 
    - Backdoor Effect Analysis: Since we are backdooring the input, we show L2 distances between the clean and the backdoored input.

The implementation includes an example usage with metrics section that: 
- Generates synthetic data.
- Trains a `RandomReLU Network` and a `BackdoorRandomReLUNetwork` using the same threshold for comparability.
- A `ModelComparator` instance to compute metrics across different levels of backdoor activation.

## Things to generalize:
- Verify Page 48 of the paper.