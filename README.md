# soap

SOAP (Synthetic Operational Aircraft Performance) is an open model for generating datasets to test aircraft performance monitoring (APM) programs. 

APM systems identify aircraft with excessive fuel consumption due to engine deterioration and increased airframe drag. The developers of APM systems face unique challenges in training and testing their models. Manufacturers or operators usually control access to the required data and may be unwilling to share proprietary data. Furthermore, even when data are available, the volume may be insufficient, and the variety of operational conditions may be limited. However, the greatest challenge to APM developers lies in evaluating variations in thrust and drag, which are not directly measurable during flight and must be inferred. This fundamental limitation of real data makes direct verification of an APM model’s thrust and drag estimates impossible.

SOAP is based on established physical formulations like those used by manufacturers for their baseline aircraft performance models. Explicit variations in drag and thrust efficiency provide ground truth for assessing model accuracy. The degree of variability in operational conditions, such as weight, altitude, ambient temperature, and fuel heating value, including seasonal effects, can be set based on the user’s expectations. These operational effects, which obscure the model’s predictive capability, can be varied to evaluate their effect on model performance. Finally, the volume of data can be adjusted by changing the virtual fleet size or the study period and is not limited.

The basis for the model is documented formally. For more information, see the following paper on https://innovate.ieee.org/techrxiv/:  L. V. Bays and M. W. Grenn, “An Open Synthetic Operational Aircraft Performance Model (SOAP),” May 2025.

