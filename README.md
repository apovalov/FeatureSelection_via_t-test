# Sequential forward selection (SFS)

It is a greedy iterative feature selection algorithm that starts with an empty feature set and sequentially adds features that maximize the quality of the model. Each new feature is added only if it improves the quality of the model according to a given metric.

![Alt text](image.png)

```
SequentialFeatureSelection:
  Set included to {}
  Set excluded to {1, 2, 3, ..., N}

  For each i in range(max_features):
    Do OptimizationStep

  Return included


OptimizationStep:
  For each feature k in excluded:
    Evaluate included + {k}
    Save k's score

  Take k with the highest score
  Add k to included
  Remove k from excluded

  Return included and excluded
  ```

There is also a reverse sequential backward selection (SBS) algorithm in which features are sequentially removed.

# Baseline Model

We will work with a synthetic dataset and solve a regression task. As the ML algorithm, we use linear regression. The metric of choice is the coefficient of determination \( R^2 \).

The dataset mentioned below consists of 50-100 features. Only five of them are useful. The rest of the features are noise. We do not know which features are useful and which are not, as well as how many features there are.

We use a 3x10 cross-validation scheme (we split the dataset into three folds and repeat this operation 10 times with different `random_state`). This particular advantage of such a validation scheme before the usual single splitting into training and test sets is that it allows us to reliably evaluate the model, avoiding the randomness of a single split.


When creating the class, it accepts the following arguments:

- `model` — the model (in our case, `LinearRegression`, but it can be any other algorithm);
- `cv` — cross-validation scheme (in our case, it's an instance of the class `RepeatedKFold`);
- `max_features` — the maximum number of features to select;
- `verbose` — silent (0) or verbose (>1) execution. It's not a mandatory parameter, but it's highly recommended to implement it using the `tqdm` module for a progress bar during execution.

The class has two methods:

- `fit` — accepts a dataset with features `X` and targets `y`. It selects features and saves them to the class attribute `selected_features_`.
- `transform` — accepts a dataset `X` and returns it, only with the selected features `selected_features_`.

Class attributes:

- `n_features_` — the original number of features in the dataset `X`, which is passed to the `fit` method;
- `selected_features_` — features selected during the execution of the `fit` method;
- `n_selected_features_` — the number of selected features. Implement it as a property with the `@property` decorator.


# Statistical Criterion t-Test

To evaluate whether the metric value significantly increases or not, we use the statistical t-test for related (dependent) samples. It is implemented in the `scipy` library ([link to documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)).

We will estimate the p-value at the significance level `alpha=0.05`. In 5% of cases, we allow the "no effect" where there is none.

It is only important for us that the metric statistically significantly increases with the addition of the feature we are evaluating. Therefore, we will use a one-sided criterion. The t-test implementation in the library has a parameter that allows using the one-sided criterion. Think about what value of the parameter correctly indicates this.

## Changes in the SFS Algorithm

The following change is introduced in the SFS algorithm. On each iteration of adding the next feature:

- If the p-value is less than the significance level `alpha`, then we add the feature to the list of candidates for addition.
- At the end of the iteration, we select the feature from all candidates that most significantly increases the metric value. Here we use the average metric value across folds.


# Lasso Regression

Excellent! You have implemented the Sequential Forward Selector algorithm, added a t-test to it, and a Bonferroni correction. This is quite a reliable but computationally expensive algorithm. The latter depends on the model and dataset.

**Note**
You can find ready-made implementations of the SFS, SBS, SBFS, SFFS algorithms. An example of one of the SFFS implementations can be found in the recommended links at the end of the page.

Let's consider another way of selecting useful features. This is **linear regression with L1 regularization (Lasso)**. As is well known, L1 regularization allows shrinking coefficients for those features that do not contribute positively to the prediction of the target. The amount of regularization applied is determined by the regularization parameter. The larger it is, the more features will be selected. This property is used for feature selection.

# Recommended Reading

1. [Sequential Forward Feature Selection (SFFS)](https://museotoolbox.readthedocs.io/en/latest/auto_examples/ai/SFFS.html#sphx-glr-auto-examples-ai-sffs-py)
2. [Feature selection in machine learning using Lasso regression](https://towardsdatascience.com/feature-selection-in-machine-learning-using-lasso-regression-7809c7c2771a)
