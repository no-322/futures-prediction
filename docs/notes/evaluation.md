# ESL Chapter7

## Bias, Variance and model complexity

First let's define our errors. We have 3 different errors:

1. *Test error or generalization error:*

$$
\mathrm{Err}_{\tau} = \mathbb{E}\left[ L\!\left(Y,\hat{f}(X)\right) \mid \tau \right]
$$

This is the error given the training data

2. *Expected test error or expected prediction error:*

$$
\mathrm{Err} = \mathbb{E}\left[L\!\left(Y,\hat{f}(X)\right)\right] = \mathbb{E}\left[\mathrm{Err}_{\tau}\right]
$$

This is what we get if we take the expectation over all possible training sets. This is not what we are interested in but this is easier to calculate and handle.

3. *Training error:*

$$
\overline{\mathrm{err}} = \frac{1}{N} \sum_{i=1}^{N} L\!\left(y_i,\hat{f}(x_i)\right)
$$

**Note:** We can not write test error (the population error) as sum/ N because that happens only when N tends to infinity (Law of Large Numbers)

### Deviance Loss:

For classification problems error can be defined in two ways: solely based on the output or based on predicted probabilities of the output ie

$$
L\!\left(G,\hat{G}(X)\right) = I\!\left(G \neq \hat{G}(X)\right)
$$

$$
L\!\left(G,\hat{p}(X)\right)
=
-2 \sum_{k=1}^{K} I(G = k)\log \hat{p}_k(X)
=
-2 \log \hat{p}_G(X)
$$

The second error is called the *deviance loss* . It is -2 times the log likelihood. This is better because it has smooth gradients and punishes models that are incorrect and confident about it.

We can parameterize this probability by a function of X to generalize.

$$
L\!\left(Y,\theta(X)\right) = -2 \cdot \log \Pr_{\theta(X)}(Y)
$$

The -2 exists so that it would make it log-likelihood loss and we can use it directly for statistical tests

$$
\begin{array}{|l|l|l|}
\hline
\textbf{Model Family} & \textbf{Distribution} & \boldsymbol{\theta(X)} \\
\hline
\text{Linear regression} &
\mathcal{N}(\mu,\sigma^2) &
\theta(X)=f(X)\ \text{(the mean)} \\
\hline
\text{Logistic regression} &
\text{Bernoulli}(p) &
\theta(X)=p(X)=\sigma(\beta^\top X) \\
\hline
\text{Multinomial classification} &
\text{Categorical}(p_1,\ldots,p_K) &
\theta(X)=\left(\hat{p}_1(X),\ldots,\hat{p}_K(X)\right) \\
\hline
\text{Poisson regression} &
\text{Poisson}(\lambda) &
\theta(X)=\lambda(X)=e^{\beta^\top X} \\
\hline
\text{GARCH-style vol model} &
\text{Gaussian with varying variance} &
\theta(X)=\sigma^2(X) \\
\hline
\end{array}
$$

In classification problems Pr(Y) is clearly probability. For regression it is the density but the intuition regarding penalizing is the same.

**Eg** for linear regression we would use f(X)= mean given by regression formula and substitute in Normal distribution pdf.

___

We have 2 goals during evaluation:

1. *Model Selection:* Select the best model: the one giving the best fit
2. *Model Assessment:* Estimate the prediction error.

If we have inf data this becomes easy. Split it as 50%, 25%, 25% to do training, validation (model selection) and test(model assessment). A nuanced split would depend on the signal:noise ratio in the data.

## Error Decomposition:

Assuming the following:

$$
\begin{aligned}
Y &= f(X) + \varepsilon \\
\mathbb{E}(\varepsilon) &= 0 \\
\mathrm{Var}(\varepsilon) &= \sigma^2_\varepsilon
\end{aligned}
$$

for a regression fit $\hat{f}(x)$ at any given point $x_0$:

$$
\mathrm{Err}(x_0)
=
\mathbb{E}\left[\left(Y - \hat{f}(x_0)\right)^2 \mid X = x_0\right]
=
\sigma_\varepsilon^2
+
\mathrm{Bias}^2\!\left(\hat{f}(x_0)\right)
+
\mathrm{Var}\!\left(\hat{f}(x_0)\right)
$$

$$
=
\text{Irreducible Error}
+
\text{Bias}^2
+
\text{Variance}
$$

* The $\sigma^2_\varepsilon$ can not be avoided. It is due to randomness in noise.
* The bias comes from how the average of our estimate differs from the true f(X)
* Variance is the variance of $\hat{f}(x)$

Typically, with increase in complexity bias goes down but variance increases.

For all models: *Bias = Model Bias + Estimation Bias*

Model Bias here refers to the bias by the best fit of the given model

$$
f^* = \arg\min_{g \in \mathcal{M}} \mathbb{E}\left[(Y - g(X))^2\right]
$$

* Bias = $f(x_0) - \mathbb{E}\hat{f}(x_0)$
* Model Bias = $f(x_0) - f^*(x_0)$
* Estimation Bias = $f^*(x_0) - \mathbb{E}\hat{f}(x_0)$

For linear models:

$$
\mathbb{E}_{x_0}\!\left[\mathrm{Bias}^2\right]
=
\mathbb{E}_{x_0}\!\left[\mathrm{Model\ Bias}^2\right]
+
\mathbb{E}_{x_0}\!\left[\mathrm{Estimation\ Bias}^2\right]
$$

Due to L2 orthogonality. The model bias removes the linear components of x and estimation bias is only linear component. Estimation bias is 0 for OLS, but restricted models like ridge and lasso have an estimation bias.

The entire point of choosing these models is to introduce model bias but bring down the MSE by reducing the Variance

<u>Note:</u>

$bias^2$ and $variance^2$ are not exactly exactly additive for classification. This relationship is valid for only square error. For classification when we use deviance then it is close but when we use 0/1 error all hell breaks lose. Bias and variance can be helpful depending on how they push the probabilities. Lets say actual probability of getting a class is 0.9 but our model has expected probability of 0.6. In case of 2 classes: this bias has no effect because p = 0.6 lands us at the same spot p=0.9 does.

## Optimism of training error:

We train by minimizing the loss function which is $bias^2 + variance^2$ . So the training data error is going to be less than the test error. For $\tau = \{(x_1,y_1),(x_2,y_2),\ldots,(x_N ,y_N )\}$:

1. Generalization error:

$$
\mathrm{Err}_{\mathcal{T}}
=
\mathbb{E}_{X_0,Y_0}
\left[
L\!\left(Y_0,\hat{f}(X_0)\right)
\mid \mathcal{T}
\right]
$$

Where $X_0, Y_0$ are points from the test set

2. Expected test error:

$$
\mathrm{Err}
=
\mathbb{E}_{\mathcal{T}}
\mathbb{E}_{X_0,Y_0}
\left[
L\!\left(Y_0,\hat{f}(X_0)\right)
\mid \mathcal{T}
\right]
$$

3. Training error:

$$
\overline{\mathrm{err}}
=
\frac{1}{N}
\sum_{i=1}^{N}
L\!\left(y_i,\hat{f}(x_i)\right)
$$

Test error is calculated on *out-of-sample data*. To compare training error with the test error, we first define the *in-sample error*:

$$
\mathrm{Err}_{\mathrm{in}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{E}_{Y_i^0}
\left[
L\!\left(Y_i^0,\hat{f}(x_i)\right)
\mid \mathcal{T}
\right]
$$

Here we do not change the training X's but we take expectation across all the possible Y's. This is going to be greater than $\overline{err}$ . We define this difference as *optimism* .

$$
\mathrm{op} \equiv \mathrm{Err}_{\mathrm{in}} - \overline{\mathrm{err}}
$$

Similar to our generalization error this error is not calculable. Hence we take an average over y (training x is fixed) to get expectation of optimism:

$$
\omega \equiv \mathbb{E}_Y(\mathrm{op})
$$

**Note:** the difference between taking expectation across $Y$ and $Y^0$ is that although both are making Y the random variable.. when we do with Y.. the error is still correlated to $\hat{f}$ but when we take across $Y^0$ there is no correlation.

$$
\omega
=
\frac{2}{N}
\sum_{i=1}^{N}
\mathrm{Cov}\!\left(\hat{y}_i, y_i\right)
$$

This means the more the prediction is correlated to the $y_i$ the more optimism exists (ie lower error in training data)

For linear fit models that use d features:

$$
\sum_{i=1}^{N}
\mathrm{Cov}\!\left(\hat{y}_i, y_i\right)
=
d\sigma_\varepsilon^2
$$

$$
\mathbb{E}_Y\!\left(\mathrm{Err}_{\mathrm{in}}\right)
=
\mathbb{E}_Y\!\left(\overline{\mathrm{err}}\right)
+
2 \cdot \frac{d}{N}\sigma_\varepsilon^2
$$

Notice how this is similar to R-squared. With increase in dimensions both R-sq and optimism increase.

In-sample error is not very useful because future values need not be same as training data. But they are important and can be used for model-selection.

___

## Estimating In-Sample Prediction Error: