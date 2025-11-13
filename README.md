# Diagnosing Long-Term Slight Overcharging in Lithium-Ion Batteries via Novel Electrochemical Features

This repository contains the code and analysis for the paper, "Diagnosing Long-Term Slight Overcharging in Lithium-Ion Batteries via Novel Electrochemical Features."

![Framework-Graphic](ToC.jpg)

---

## Core Framework

This project employs a **feature engineering** approach to extract physically meaningful Health Indicators (HIs) from standard operational data. These HIs serve as a compact and interpretable representation of the battery's electrochemical state, which can then be fed into a simple classifier.

1.  **Problem:** Detecting and isolating long-term slight overcharging (e.g., 4.35V, 4.5V) from normal operation (4.2V) in 18650 NMC cells.
2.  **Feature Engineering:** We extract a set of HIs for each cycle, including:
    * **Voltage-Based:** Starting Voltage, Knee Voltage, and Mean Voltage Derivative.
    * **ICA-Based:** Positions and magnitudes of key peaks and valleys, which correspond to fundamental electrochemical phase transitions ($H1 \leftrightarrow M$, $M \leftrightarrow H2$).
3.  **Key Diagnostic Feature:** The algorithm is designed to detect the emergence of the $H2 \leftrightarrow H3$ phase transition, which is a clear structural and electrochemical marker of high-voltage overcharge.
4.  **Classification:** A lightweight Multi-Layer Perceptron (MLP) is trained on this feature vector to classify each cycle as "Healthy," "Mild Fault," or "Severe Fault."

This approach allows the model to achieve high accuracy (98.68%) while remaining highly interpretable, as shown by the permutation feature importance analysis.

---