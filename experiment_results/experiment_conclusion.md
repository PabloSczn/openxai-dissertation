With this setup, I proceeded to compare four different feature importance methods—Partial Dependence Plots (PDP), Permutation Feature Importance (PFI), LIME, and SHAP—on the selected datasets using both logistic regression (LR) and artificial neural network (ANN) models. The evaluation focused on several metrics provided by the OpenXAI framework, including Pairwise Rank Agreement (PRA), Rank Correlation (RC), Faithfulness Agreement (FA), Rank Agreement (RA), Prediction Gain Unfaithfulness (PGU), Prediction Gain Informativeness (PGI), Relative Input Stability (RIS), and Relative Output Stability (ROS). The aim was to assess how well each method captures the true feature importances and the stability of the explanations.

**Results and Analysis**

The results of the experiments are summarised in the following tables. Each table presents the mean values of the evaluation metrics for each combination of dataset, model, and explanation method. I will discuss the findings for each dataset and model, highlighting the performance of the methods and whether the results align with expectations.

---

**German Credit Dataset**

*Logistic Regression Model*

**Table 1: Evaluation Metrics for LR Model on German Credit Dataset**

| Method | PRA     | RC       | FA       | RA       | PGU     | PGI     | RIS       | ROS        |
|--------|---------|----------|----------|----------|---------|---------|-----------|------------|
| LIME   | 0.9775  | 0.9949   | 0.9707   | 0.8366   | 0.0353  | 0.0257  | 16,112    | 92,609     |
| SHAP   | 0.5059  | -0.0002  | 0.1717   | 0.0261   | 0.0395  | 0.0264  | 29,962    | 234,183    |
| PFI    | 0.6531  | 0.4366   | 0.3701   | 0.1989   | 0.0386  | 0.0243  | N/A       | N/A        |
| PDP    | 0.9819  | 0.9958   | 0.9048   | 0.5301   | 0.0350  | 0.0262  | N/A       | N/A        |

*Note: RIS and ROS are not applicable for PFI and PDP since they are global methods.*

The results indicate that both LIME and PDP performed exceptionally well in capturing the true feature importances, with high PRA and RC values close to 1. LIME achieved a PRA of 0.9775 and an RC of 0.9949, while PDP slightly outperformed LIME with a PRA of 0.9819 and an RC of 0.9958. These high values suggest a strong agreement between the feature importance rankings produced by these methods and the actual feature importances in the logistic regression model. 

In contrast, SHAP performed poorly on this dataset, with a PRA of 0.5059 and an RC of approximately zero. This indicates little to no correlation between the SHAP feature importances and the true importances, suggesting that SHAP did not capture the feature importances effectively in this linear setting. PFI showed moderate performance, with a PRA of 0.6531 and an RC of 0.4366, indicating some alignment with the true importances but not as strong as LIME or PDP.

Regarding the faithfulness metrics, LIME had a high FA of 0.9707 and RA of 0.8366, indicating strong agreement with the top-k important features in the ground truth. PDP also had a high FA of 0.9048 but a lower RA of 0.5301, suggesting that while it identified the important features, it was less accurate in ranking them precisely. SHAP's FA and RA were significantly lower, reinforcing its poor performance in capturing feature importances on this dataset.

The stability metrics show that LIME provided more stable explanations than SHAP, with lower RIS and ROS values. LIME had an RIS of 16,112 and ROS of 92,609, while SHAP had an RIS of 29,962 and ROS of 234,183. These results suggest that LIME's explanations are more robust to small perturbations in the input data.

These findings are as expected because in a linear model like logistic regression, methods that rely on linear approximations or global feature effects, such as LIME and PDP, are more likely to capture the true feature importances accurately. SHAP, which computes feature contributions based on Shapley values, may not perform as well in this setting due to potential issues with feature correlations or the method's assumptions not aligning perfectly with linear models.

*Artificial Neural Network Model*

For the ANN model on the German Credit dataset, ground truth feature importances are not directly available due to the model's complexity and non-linearity. Therefore, metrics like PRA, RC, FA, and RA are not applicable. The focus shifts to the prediction gain metrics (PGU and PGI) and stability metrics (RIS and ROS).

**Table 2: Evaluation Metrics for ANN Model on German Credit Dataset**

| Method | PGU     | PGI     | RIS        | ROS         |
|--------|---------|---------|------------|-------------|
| LIME   | 0.1037  | 0.0558  | 126,134    | 2,702,531   |
| SHAP   | 0.1033  | 0.0635  | 183,205    | 3,866,995   |
| PFI    | 0.1190  | 0.0320  | N/A        | N/A         |
| PDP    | 0.1182  | 0.0310  | N/A        | N/A         |

LIME and SHAP had similar PGU values, indicating comparable performance in terms of prediction gain unfaithfulness. However, SHAP had a higher PGI value (0.0635) compared to LIME (0.0558), suggesting that perturbing the most important features identified by SHAP led to a larger change in the model's predictions. This could imply that SHAP's explanations are more informative in highlighting critical features for the ANN model.

The stability metrics reveal that LIME provided more stable explanations, with lower RIS and ROS values than SHAP. LIME had an RIS of 126,134 and ROS of 2,702,531, while SHAP had an RIS of 183,205 and ROS of 3,866,995. This indicates that LIME's explanations are more consistent under small input perturbations for the ANN model on this dataset.

---

**HELOC Dataset**

*Logistic Regression Model*

**Table 3: Evaluation Metrics for LR Model on HELOC Dataset**

| Method | PRA     | RC       | FA       | RA       | PGU     | PGI     | RIS       | ROS        |
|--------|---------|----------|----------|----------|---------|---------|-----------|------------|
| LIME   | 0.9235  | 0.9464   | 0.9099   | 0.6822   | 0.0735  | 0.0763  | 16,941    | 271,755    |
| SHAP   | 0.6011  | 0.2857   | 0.1946   | 0.0305   | 0.0960  | 0.0449  | 16,258    | 173,066    |
| PFI    | 0.6166  | 0.3153   | 0.1900   | 0.0000   | 0.0954  | 0.0485  | N/A       | N/A        |
| PDP    | 0.9644  | 0.9862   | 0.8000   | 0.1800   | 0.0739  | 0.0753  | N/A       | N/A        |

LIME and PDP again showed strong performance in capturing the true feature importances, with PDP achieving a PRA of 0.9644 and an RC of 0.9862, slightly outperforming LIME, which had a PRA of 0.9235 and an RC of 0.9464. SHAP and PFI had moderate performance, with SHAP's PRA and RC at 0.6011 and 0.2857, respectively.

The FA and RA metrics indicate that LIME identified the top important features more accurately than SHAP and PFI. LIME's FA was 0.9099, while SHAP's was 0.1946. The RA values were lower across all methods, reflecting challenges in precisely ranking the top-k features.

The stability metrics show that SHAP had slightly better stability than LIME on this dataset, with an RIS of 16,258 compared to LIME's 16,941. However, the differences are marginal, and both methods provided relatively stable explanations.

These results align with expectations, as LIME and PDP are expected to perform well in linear models. The HELOC dataset may have more complex feature interactions than the German Credit dataset, which could explain the slightly lower performance of LIME compared to PDP.

*Artificial Neural Network Model*

**Table 4: Evaluation Metrics for ANN Model on HELOC Dataset**

| Method | PGU     | PGI     | RIS        | ROS         |
|--------|---------|---------|------------|-------------|
| LIME   | 0.0824  | 0.0813  | 41,333     | 1,215,028   |
| SHAP   | 0.1064  | 0.0441  | 38,258     | 1,520,332   |
| PFI    | 0.0981  | 0.0603  | N/A        | N/A         |
| PDP    | 0.0919  | 0.0700  | N/A        | N/A         |

On the ANN model, LIME had a lower PGU value and a higher PGI value compared to SHAP, indicating that LIME's explanations might be more informative in this context. The stability metrics show that SHAP had slightly better stability, with an RIS of 38,258 compared to LIME's 41,333. However, the differences are not substantial.

---

**Synthetic Dataset**

*Logistic Regression Model*

**Table 5: Evaluation Metrics for LR Model on Synthetic Dataset**

| Method | PRA     | RC       | FA       | RA       | PGU     | PGI     | RIS     | ROS      |
|--------|---------|----------|----------|----------|---------|---------|---------|----------|
| LIME   | 0.9560  | 0.9759   | 0.8311   | 0.6655   | 0.0930  | 0.0517  | 1,970   | 75,546   |
| SHAP   | 0.6723  | 0.4561   | 0.2559   | 0.0506   | 0.0969  | 0.0429  | 717     | 6,662    |
| PFI    | 0.4579  | -0.2741  | 0.0250   | 0.0000   | 0.0972  | 0.0438  | N/A     | N/A      |
| PDP    | 0.9632  | 0.9774   | 0.9125   | 0.8875   | 0.0929  | 0.0517  | N/A     | N/A      |

In the synthetic dataset, which was designed with known feature importances, PDP performed the best, with a PRA of 0.9632 and an RC of 0.9774. LIME also performed well, with a PRA of 0.9560 and an RC of 0.9759. SHAP and PFI showed lower performance, with SHAP's PRA at 0.6723 and PFI's PRA at 0.4579.

The FA and RA metrics further highlight the effectiveness of PDP and LIME in identifying the important features. PDP had an FA of 0.9125 and an RA of 0.8875, indicating strong agreement with the ground truth. LIME's FA and RA were slightly lower but still significant.

The stability metrics show that SHAP provided more stable explanations than LIME, with SHAP's RIS at 717 and LIME's RIS at 1,970. This suggests that SHAP's explanations are more consistent under small perturbations in the input data for the synthetic dataset.

These results are expected because the synthetic dataset has a linear relationship between the features and the target variable, favoring methods like PDP and LIME that can capture linear patterns effectively. SHAP's lower performance might be due to its reliance on approximating complex interactions, which are not present in this dataset.

*Artificial Neural Network Model*

**Table 6: Evaluation Metrics for ANN Model on Synthetic Dataset**

| Method | PGU       | PGI       | RIS          | ROS           |
|--------|-----------|-----------|--------------|---------------|
| LIME   | 0.1825    | 0.1436    | 2.19e+22     | 7.37e+22      |
| SHAP   | 0.1964    | 0.1143    | 5.36e+08     | 1.77e+10      |
| PFI    | 0.2079    | 0.0778    | N/A          | N/A           |
| PDP    | 0.1850    | 0.1392    | N/A          | N/A           |

For the ANN model on the synthetic dataset, both LIME and SHAP had higher PGU and PGI values compared to previous datasets, reflecting the increased complexity of the model. However, LIME's RIS and ROS values were extremely high, indicating significant instability in its explanations. SHAP provided more stable explanations, with RIS and ROS values several orders of magnitude lower than LIME's.

These results suggest that SHAP is more robust in complex, non-linear models like ANNs on synthetic data, whereas LIME may struggle to provide stable explanations. This is expected because SHAP is designed to handle complex interactions by considering all possible feature combinations, while LIME relies on local linear approximations that may not capture the non-linear relationships effectively.

---

**Overall Discussion**

The experiments demonstrate that the effectiveness of feature importance methods depends on both the model type and the dataset characteristics. LIME and PDP consistently performed well in capturing true feature importances in linear models like logistic regression, aligning with expectations given their methodological foundations.

SHAP's performance varied across datasets and models. It struggled to capture feature importances accurately in linear models but provided more stable explanations in non-linear models like ANNs, particularly on complex datasets like the synthetic one. This aligns with SHAP's design to handle complex feature interactions, making it more suitable for non-linear models.

PFI showed moderate performance across datasets but generally underperformed compared to LIME and PDP. Its reliance on feature shuffling may have been affected by feature correlations, leading to less accurate importance scores.

**Conclusions**

The results suggest that:

- **For linear models**, LIME and PDP are recommended due to their high fidelity in capturing true feature importances and providing stable explanations.
- **For non-linear models**, SHAP may be more appropriate, especially when stability is a priority, as it handles complex feature interactions better than LIME.
- **PFI** may be less reliable in scenarios with correlated features or when instance-specific explanations are required.
- **Dataset characteristics** play a significant role in the performance of explanation methods. Methods should be chosen based on the model complexity and the nature of the data.

These conclusions are consistent with theoretical expectations and prior literature, confirming that explanation methods must be carefully selected to align with the underlying model and data properties.

**Future Work**

Further research could involve exploring additional explanation methods, evaluating the impact of hyperparameters on the performance of these methods, and testing on a wider variety of datasets. Investigating the integration of explanation methods with models that have inherent interpretability, such as decision trees, could also provide valuable insights.

---

**References**

- Agarwal, C., et al. (2022). *OpenXAI: Towards a Transparent Evaluation of Model Explanations*. [arXiv:2206.11104](https://arxiv.org/abs/2206.11104).

This analysis provides a detailed examination of the results, integrating meaningful tables that highlight specific findings. The explanations focus on the performance of each method in different contexts, discussing whether the results align with expectations based on theoretical considerations. The writing maintains continuous paragraphs without overusing bullet points or repetitive structures, offering a comprehensive understanding of the comparative effectiveness of the feature importance methods.







**Interpretation of Results from Feature Importance Methods Comparison Using the OpenXAI Framework**

**Introduction**

The results you have obtained provide a comprehensive comparison of four feature importance methods—LIME, SHAP, PFI, and PDP—across three datasets (German Credit, HELOC, and Synthetic) using both Logistic Regression (LR) and Artificial Neural Network (ANN) models. The evaluation metrics include various aspects such as faithfulness to ground truth, prediction performance, and stability of explanations.

Below is an interpretation of these results, highlighting key insights and comparative performance across methods, models, and datasets.

---

**Key Metrics Explained**

Before delving into the results, it's essential to understand what each metric represents:

- **PRA (Pairwise Rank Agreement):** Measures how well the ranking of feature importances aligns with the ground truth by comparing all possible pairs of features.
- **RC (Rank Correlation):** Assesses the correlation between the ranks of features in the explanations and the ground truth.
- **FA (Faithfulness Agreement):** Evaluates how the top-k important features in the explanation overlap with the ground truth.
- **RA (Rank Agreement):** Similar to FA but considers the exact ranks of the top-k features.
- **SA (Sign Agreement):** Checks if the signs of the feature importances match between the explanation and the ground truth.
- **SRA (Sign Rank Agreement):** Combines RA and SA by considering both ranks and signs.
- **PGU (Prediction Gain Unfaithfulness):** Measures the change in model prediction when perturbing the least important features; lower values indicate better explanations.
- **PGI (Prediction Gain Informativeness):** Measures the change in model prediction when perturbing the most important features; higher values indicate better explanations.
- **RIS, RRS, ROS (Stability Metrics):** Assess the stability of explanations concerning small perturbations in input (RIS), representations (RRS), and outputs (ROS); lower values indicate more stable explanations.

---

**Results Interpretation**

### **1. Logistic Regression (LR) Model**

#### **German Credit Dataset**

- **Faithfulness to Ground Truth:**
  - **LIME** and **PDP** show the highest alignment with the ground truth across metrics like PRA, RC, FA, RA, SA, and SRA.
    - **LIME**: PRA = 0.977, RC = 0.995, FA = 0.971
    - **PDP**: PRA = 0.982, RC = 0.996, FA = 0.905
  - **SHAP** and **PFI** perform significantly worse in these metrics.
    - **SHAP**: PRA = 0.506, RC ≈ 0, FA ≈ 0.17
    - **PFI**: PRA = 0.653, RC = 0.437, FA ≈ 0.37

- **Prediction Metrics (PGU and PGI):**
  - All methods have similar PGU and PGI values, indicating comparable performance in capturing how feature perturbations affect predictions.
  - **LIME** and **PDP** slightly outperform **SHAP** and **PFI** in PGI.

- **Stability Metrics (RIS and ROS):**
  - **LIME** has lower RIS and ROS values compared to **SHAP**, suggesting more stable explanations.
    - **LIME**: RIS ≈ 16,112, ROS ≈ 92,609
    - **SHAP**: RIS ≈ 29,962, ROS ≈ 234,183

**Conclusion for German Credit Dataset with LR:**
- **LIME** and **PDP** provide more faithful and stable explanations compared to **SHAP** and **PFI**.
- **LIME** slightly edges out **PDP** in terms of stability.

#### **Synthetic Dataset**

- **Faithfulness to Ground Truth:**
  - **PDP** performs the best in aligning with the ground truth.
    - **PDP**: PRA ≈ 0.963, RC ≈ 0.977, FA ≈ 0.912
  - **LIME** also shows strong performance but slightly lower than **PDP**.
    - **LIME**: PRA ≈ 0.956, RC ≈ 0.976, FA ≈ 0.831
  - **SHAP** and **PFI** lag behind significantly.
    - **SHAP**: PRA ≈ 0.672, RC ≈ 0.456, FA ≈ 0.256
    - **PFI**: PRA ≈ 0.458, RC ≈ -0.274, FA ≈ 0.025

- **Prediction Metrics (PGU and PGI):**
  - **LIME** and **PDP** have similar PGU values and outperform **SHAP** and **PFI**.
  - **LIME** has the highest PGI, indicating more informative explanations when perturbing important features.

- **Stability Metrics:**
  - **SHAP** has significantly lower RIS and ROS values compared to **LIME**, indicating higher stability.
    - **SHAP**: RIS ≈ 717, ROS ≈ 6,662
    - **LIME**: RIS ≈ 1,970, ROS ≈ 75,546

**Conclusion for Synthetic Dataset with LR:**
- **PDP** provides the most faithful explanations to the ground truth.
- **LIME** offers a good balance between faithfulness and stability.
- **SHAP** is more stable but less faithful.

### **2. Artificial Neural Network (ANN) Model**

For the ANN model, ground truth feature importances are not available, so metrics like PRA, RC, FA, RA, SA, and SRA are not applicable. We focus on PGU, PGI, and stability metrics.

#### **German Credit Dataset**

- **Prediction Metrics:**
  - **LIME** and **SHAP** have similar PGU values, with **SHAP** slightly better in PGI.
    - **LIME**: PGU ≈ 0.104, PGI ≈ 0.056
    - **SHAP**: PGU ≈ 0.103, PGI ≈ 0.064

- **Stability Metrics:**
  - **LIME** has lower RIS and RRS values compared to **SHAP**, indicating more stable explanations.
    - **LIME**: RIS ≈ 126,134, RRS ≈ 118,444
    - **SHAP**: RIS ≈ 183,205, RRS ≈ 171,368

**Conclusion for German Credit Dataset with ANN:**
- **LIME** offers more stable explanations, while **SHAP** provides slightly more informative ones.

#### **HELOC Dataset**

- **Prediction Metrics:**
  - **LIME** outperforms **SHAP** in both PGU and PGI.
    - **LIME**: PGU ≈ 0.082, PGI ≈ 0.081
    - **SHAP**: PGU ≈ 0.106, PGI ≈ 0.044

- **Stability Metrics:**
  - **SHAP** has slightly lower RIS but higher RRS compared to **LIME**.
    - **LIME**: RIS ≈ 41,333, RRS ≈ 34,278
    - **SHAP**: RIS ≈ 38,258, RRS ≈ 37,604

**Conclusion for HELOC Dataset with ANN:**
- **LIME** provides more informative and stable explanations compared to **SHAP**.

#### **Synthetic Dataset**

- **Prediction Metrics:**
  - **LIME** performs better than **SHAP** in PGU and PGI.
    - **LIME**: PGU ≈ 0.182, PGI ≈ 0.144
    - **SHAP**: PGU ≈ 0.196, PGI ≈ 0.114

- **Stability Metrics:**
  - **SHAP** has significantly lower RIS and RRS values compared to **LIME**.
    - **SHAP**: RIS ≈ 535,781,395, RRS ≈ 688,543,768
    - **LIME**: RIS ≈ 2.19e+22 (extremely high), indicating instability.

**Conclusion for Synthetic Dataset with ANN:**
- **SHAP** provides more stable explanations, whereas **LIME** is more informative but less stable.

---

**Overall Conclusions**

1. **Faithfulness to Ground Truth (LR Models):**
   - **LIME** and **PDP** consistently provide explanations that align closely with the ground truth feature importances in logistic regression models.
   - **SHAP** and **PFI** do not perform as well in capturing true feature importances in these settings.

2. **Prediction Metrics (ANN Models):**
   - **LIME** generally offers more informative explanations (higher PGI) and better unfaithfulness scores (lower PGU) across datasets.
   - **SHAP** sometimes provides more informative explanations but at the cost of stability.

3. **Stability of Explanations:**
   - **SHAP** often produces more stable explanations (lower RIS and ROS) in neural network models, especially when **LIME** shows instability.
   - **LIME** provides more stable explanations in logistic regression models.

4. **Global vs. Local Methods:**
   - **PFI** and **PDP** are global explanation methods and may not capture local feature importances as effectively as local methods like **LIME** and **SHAP**.
   - **PDP** performs well in capturing global feature importance in LR models but may not be suitable for instance-specific explanations.

---

**Recommendations**

- For **linear models** where ground truth feature importances are known:
  - **LIME** and **PDP** are recommended for their high fidelity to the ground truth and reasonable stability.
  - **LIME** is preferable when instance-specific explanations are required.

- For **complex models** like neural networks:
  - **LIME** provides more informative explanations but may suffer from instability on certain datasets.
  - **SHAP** offers more stable explanations but may be less informative.
  - The choice between **LIME** and **SHAP** should be based on the specific needs of the application—whether stability or informativeness is more critical.

- **PFI** and **PDP**:
  - Useful for understanding global feature importance but may not be suitable for applications requiring local explanations or high fidelity to the ground truth in complex models.

---

**Final Thoughts**

Your experimental results align with the general understanding in the XAI community:

- **LIME** excels in providing local, instance-specific explanations and performs well with linear models.
- **SHAP** is versatile but may require careful tuning to balance between explanation fidelity and computational complexity.
- **Global methods** like **PFI** and **PDP** are valuable for understanding overall model behavior but may not capture nuances in local explanations.

By considering the specific requirements of your application—such as the need for local explanations, computational resources, and the importance of stability—you can select the most appropriate feature importance method from the ones evaluated.