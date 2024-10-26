# Backdooring

In `backdoor-analysis.py` we aim to show backdooring in practice. We use ideas in Page 48 of the paper.

1. First we set a reasonable threshold $\tau$ based on the expected variance and subgaussian concentration properties of the ReLU on a gaussian.
2. We need to ensure that the backdoored inputs have a sufficiently high variance so that they exceed the above threshold $\tau$ more often than the natural case.

## Step 1: Setting threshold for Normal inputs
To find an appropriate threshold, we need to calculate the expected value and variance of the ReLU-activated features for normal inputs. The idea is to set $\tau$ so it exceeds the average ReLU output for normal inputs. Therefore the classifier would activate less than half the time due to this $\tau$.

We do this via the `set_threshold()` function inside the `RandomReLUNetwork` class. For example we set the threshold at the 90th percentile. This will ensure only about 1/10th of the time the classifier will turn 1.

Furthermore, in the `sample_random_relu()` we scaled them with a factor $1/\sqrt{\text{input_dim}}$ to give them even less variance.

## Step 2: Define the backdoored network with higher variance
For backdoored inputs we want the average feature value to exceed $\tau$ with high probability. Therefore in the `BackdoorRandomReLUNetwork` we adjust the weight generation process to incorporate the "spiked" variance due to the secret key $s$ and the parameter $\theta$, ensuring that backdoored features will have greater variance.

    - Observe that in the `sample_backdoor_relu` method we have `cov` the modified covariance matrix incorporating the "spiked" distribution.

    - This method ensures that the variance of each feature is increased by the spike from the secret key, which should result in backdoored inputs exceeding $\tau$ with high probability.

## Step 3: Test the Backdoor Activation with various $\lambda$ values
We then use the `activate_backdoor` function to add the secret key $s$ to any input $x$ to create backdoored input $x' = x + \lambda s$. The network should output a positive classification for backdoored inputs if the activation changes effectively.

## Step 4: Evaluate the Activation Success
The `ModelComparator` class can evaluate the classification accuracy on normal vs. backdoored inputs, recording whether the threshold $\tau$ is exceeded for backdoored inputs with high probability.


## Explanation of Results
1. We see that in the backdoor model a positive classification rate of 1.0 across all $\lambda$ values, showing that the backdoored inputs are reliably activated by the ReLU features, likely because of the increased mean and variance from the injected secret key.

2. In the Natural Model the positive classification rate is generally low except at $\lambda = 1.0$ suggesting that the threshold $\tau$ may be getting exceeded as the scale of the backdoor signal increases. 


## Future work
1. Adjust threshold $\tau$ for Robustness.
We may want to increase $\tau$ at higher $\lambda$ values to increases robustness. For example we may increase $\tau$ incrementally by 5-10% to see if this lowers the natural model's positive classification rate at high $\lambda$ without impacting the backdoor model.

2. Control backdoor strength via $\theta$ or Sparsity.
If needed we can decrease the sparsity of the secret key slightly or reduce $\theta$ in the spiked covariance matrix, which would lower the impact of the backdoor in the features without compromising overall behavior.

3. Confirm concentration and evaluate edge cases.
To ensure that the threshold is robustly separating normal inputs from the backdoored ones, we may want to evaluate the variance and mean of $\langle g_i, x \rangle$ and $\langle g_i , x' \rangle$ to check concentration around the mean for both cases. Furthermore we may want to  verify that the normal inputs concentrate closely around their expected mean below $\tau$ and the backdoored inputs concentrate above it, thereby confirming the effectiveness of the threshold in most situations.