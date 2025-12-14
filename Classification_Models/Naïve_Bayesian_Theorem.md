# Naïve Bayes Classification

Naïve Bayes is a probabilistic machine learning algorithm based on **Bayes’ Theorem**. It is widely used for classification tasks, particularly when working with small datasets or when fast, interpretable models are required.


## 1. Sample Data for Classification

| height_cm | weight_kg | age | activity_level | label |
|----------:|----------:|----:|---------------|--------|
| 175.0 | 86.8 | 47 | Medium | Sporty |
| 168.6 | 81.1 | 48 | Medium | Sporty |
| 176.5 | 70.7 | 55 | Medium | Sporty |
| 185.2 | 62.2 | 28 | Medium | Sporty |
| 167.7 | 78.4 | 52 | High | Non-Sporty |


## 2. Problem Definition
The goal is to predict whether an individual is **Sporty** or **Non-Sporty** based on height (cm), weight (kg), age (years) and activity level.

This is a **binary classification problem** as we can see there are two classesin the label column.


## 3. Bayes’ Theorem

Bayes’ Theorem is a core principle of probability theory that provides a formal mechanism for **updating beliefs in the presence of new evidence**. In machine learning, it forms the mathematical foundation of the Naïve Bayes classifier by enabling the calculation of the probability that a data point belongs to a particular class given its observed features.

The theorem is expressed as:

$$
P(C_k \mid X) = \frac{P(X \mid C_k)\, P(C_k)}{P(X)}
$$

Where:
- $C_k$ denotes the $k$-th class label (e.g. *Sporty* or *Non-Sporty*)
- $X = (x_1, x_2, \ldots, x_n)$ represents the observed feature vector
- $P(C_k \mid X)$ is the **posterior probability**
- $P(X \mid C_k)$ is the **likelihood**
- $P(C_k)$ is the **prior probability**
- $P(X)$ is the **evidence** or **marginal likelihood**


### 3.1 Conceptual Meaning of Bayes’ Theorem

Bayes’ Theorem describes how prior knowledge about a class is combined with evidence from observed data to produce an updated belief.

- The **prior probability** $P(C_k)$ reflects how likely class $C_k$ is *before* observing any features.  
  In the fitness dataset, the Sporty class has a higher prior probability because it appears more frequently.

- The **likelihood** $P(X \mid C_k)$ measures how probable the observed feature values are assuming the data point truly belongs to class $C_k$.

- The **posterior probability** $P(C_k \mid X)$ represents the updated probability that the data point belongs to class $C_k$ *after* considering the evidence provided by the features.

Thus, Bayes’ Theorem provides a principled way to combine **prior knowledge** with **observed data**.

### 3.2 The Evidence Term

The evidence term $P(X)$ acts as a normalising constant that ensures posterior probabilities across all classes sum to one. It is defined as:

$$
1 = \frac{\sum_{k}P(X \mid C_k)\, P(C_k)}{P(X)} 
$$
$$
∴ P(X) = \sum_{k} P(X \mid C_k)\, P(C_k)
$$

Although essential for obtaining true probability values, $P(X)$ does **not depend on the class** being evaluated. As a result, it does not influence which class is ultimately selected during classification.

### 3.3 Bayes’ Theorem in Classification Tasks

In classification problems, the objective is not to compute exact posterior probabilities, but rather to identify the class with the **highest posterior probability**. For this reason, Bayes’ Theorem is commonly rewritten in proportional form:

$$
P(C_k \mid X) \propto P(X \mid C_k)\, P(C_k)
$$

This simplification allows the classifier to compare classes directly without explicitly calculating $P(X)$.

### 3.4 Decision Rule Derived from Bayes’ Theorem

Using Bayes’ Theorem, the classification decision rule is defined as:

$$
\hat{C} = \arg\max_{C_k} P(C_k \mid X)
$$

Substituting the proportional form:

$$
\hat{C} =
\arg\max_{C_k}
\left[
P(X \mid C_k)\, P(C_k)
\right]
$$

This rule states that a data point is assigned to the class for which the combination of how common the class is (prior), and how well the features fit the class (likelihood) is maximised.

### 3.5 Role of Bayes’ Theorem in Naïve Bayes

Naïve Bayes applies Bayes’ Theorem together with a simplifying assumption: **all features are conditionally independent given the class**. Under this assumption, the likelihood term can be factorised as:

$$
P(X \mid C_k) =
\prod_{i=1}^{n} P(x_i \mid C_k)
$$

This factorisation dramatically simplifies computation and makes the algorithm efficient, even for high-dimensional data.

### 3.6 Application to the Fitness Dataset

For the fitness dataset:
- $C_k \in \{\text{Sporty}, \text{Non-Sporty}\}$
- $X$ includes height, weight, age, and activity level

Bayes’ Theorem enables the model to:
1. Start with prior beliefs about how common each class is.
2. Evaluate how likely the observed measurements are for each class.
3. Combine both sources of information into a posterior probability.
4. Select the most probable class.

This probabilistic reasoning allows Naïve Bayes to perform reliably even with very small datasets, such as the one used in this example.

### 3.7 Importance of Bayes’ Theorem in Machine Learning

Bayes’ Theorem is central to many probabilistic machine learning models because it:
- provides a clear mathematical interpretation of uncertainty,
- allows incorporation of prior knowledge,
- supports learning from limited data,
- and forms the foundation of Bayesian inference methods.

Naïve Bayes represents one of the simplest and most direct applications of Bayes’ Theorem in practical machine learning.

## 4. Naïve Independence Assumption

Naïve Bayes assumes that all features are **conditionally independent given the class**:

$$
P(X \mid C_k) =
P(x_1 \mid C_k)\,
P(x_2 \mid C_k)\,
\dots\,
P(x_n \mid C_k)
$$

In this dataset:

$$
P(X \mid C_k) =
P(\text{height} \mid C_k)\,
P(\text{weight} \mid C_k)\,
P(\text{age} \mid C_k)\,
P(\text{activity} \mid C_k)
$$

## 5 Class Prior Probabilities

The prior probability of each class is calculated from the dataset.

Total samples: $5$

- Sporty: $4$
- Non-Sporty: $1$

$$
P(\text{Sporty}) = \frac{4}{5} = 0.8
$$

$$
P(\text{Non-Sporty}) = \frac{1}{5} = 0.2
$$

## 6. Likelihood Estimation

### 6.1 Continuous Features (Gaussian Naïve Bayes)

The features **height**, **weight**, and **age** are continuous.  
Gaussian Naïve Bayes assumes that each continuous feature follows a **normal distribution** within each class.

The likelihood function is:

$$
P(x_i \mid C_k) =
\frac{1}{\sqrt{2\pi\sigma_{k,i}^2}}
\exp\left(
-\frac{(x_i - \mu_{k,i})^2}{2\sigma_{k,i}^2}
\right)
$$

Where:
- $x_i$ is the value of feature $i$ for the new data point  
- $\mu_{k,i}$ is the **mean** of feature $i$ for class $k$  
- $\sigma_{k,i}^2$ is the **variance** of feature $i$ for class $k$  

#### Example: Mean Height for the Sporty Class

Sporty heights:

$$
\{175.0,\;168.6,\;176.5,\;185.2\}
$$

Mean:

$$
\mu_{\text{height,Sporty}} =
\frac{175.0 + 168.6 + 176.5 + 185.2}{4}
= 176.33
$$

The variance $\sigma^2_{\text{height,Sporty}}$ is calculated using standard statistical formulas.

### 6.2 Categorical Feature (Activity Level)

Activity level is a **categorical** feature with two possible values:

$$
\{\text{Medium}, \text{High}\}
$$

Observed counts:

| Class | Medium | High |
|------|--------|------|
| Sporty | 4 | 0 |
| Non-Sporty | 0 | 1 |

To prevent zero probabilities, **Laplace smoothing** is applied:

$$
P(x_i \mid C_k) =
\frac{N_{k,i} + 1}{N_k + V}
$$

Where:
- $N_{k,i}$ = count of feature value $i$ in class $k$
- $N_k$ = total samples in class $k$
- $V$ = number of possible categories ($V=2$)

## 7. Posterior Probability Calculation

For a new observation $X$, the posterior probability is:

$$
P(C_k \mid X) \propto
P(C_k)
\prod_{i=1}^{n} P(x_i \mid C_k)
$$

To avoid numerical underflow, logarithms are used:

$$
\log P(C_k \mid X) =
\log P(C_k) +
\sum_{i=1}^{n} \log P(x_i \mid C_k)
$$

## 8. Classification Decision

The predicted class is the one with the maximum posterior probability:

$$
\hat{C} =
\arg\max_{C_k}
\left[
\log P(C_k) +
\sum_{i=1}^{n} \log P(x_i \mid C_k)
\right]
$$

## 9. Strengths and Limitations

### Strengths
- Simple and computationally efficient  
- Performs well with small datasets  
- Produces interpretable probabilistic outputs  

### Limitations
- Assumes feature independence  
- Sensitive to incorrect distribution assumptions  
- Limited ability to model complex relationships  

## 10. Conclusion
This example demonstrates how Gaussian Naïve Bayes can be applied to a small fitness dataset. By estimating class priors, modeling feature likelihoods, and applying Bayes’ Theorem, the algorithm predicts whether an individual is Sporty or Non-Sporty in a mathematically principled way.
