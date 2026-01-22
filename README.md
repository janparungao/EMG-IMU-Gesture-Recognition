# EMG-IMU Gesture Recognition

This project investigates upper-limb gesture recognition using a combination of surface
electromyography (EMG) and inertial measurement unit (IMU) data. The primary contribution
is an end-to-end machine learning pipeline that evaluates EMG-only, IMU-only, and fused
EMG-IMU models, highlighting the potential benefits of sensor fusion for robust gesture
classification.

This work was developed as part of an MSc-level research project and focuses on the
design, implementation, and evaluation of a complete offline gesture recognition system.

---

## Motivation

Gesture recognition is a key component of human–computer interaction, with applications
in robotics, prosthetics, rehabilitation, and assistive technologies. While EMG and IMU
signals have each been widely studied in isolation, there remains a gap in practical,
end-to-end implementations that systematically compare unimodal approaches against
feature-level sensor fusion.

This project aims to address this gap by:
- Investigating the complementary nature of EMG (muscle activation) and IMU (kinematics)
- Evaluating whether feature-level fusion improves classification performance
- Providing a reproducible pipeline from raw sensor data to trained models

---

## Sensors and Data Acquisition

Data were collected using Delsys Trigno wearable sensors, providing:
- Surface EMG signals from multiple upper-limb muscles
- Tri-axial accelerometer and gyroscope measurements

The dataset consists of time-synchronised EMG and IMU signals labelled by performed
gesture.

> ⚠️ **Note:** The full dataset is not included in this repository to adhere to University Ethics and Regulations
> data-sharing considerations.

---

## Data Processing Pipeline

The system implements a complete offline processing pipeline:

1. **Data loading**  
   Raw EMG and IMU signals are loaded from CSV format.

2. **Signal preprocessing**  
   - EMG signals are filtered and processed to obtain informative representations  
   - IMU signals are cleaned and aligned with EMG data  

3. **Windowing**  
   Continuous time-series data are segmented into overlapping temporal windows to capture
   dynamic gesture information.

4. **Feature extraction**  
   - **EMG features:** time-domain descriptors such as RMS, waveform length, and related
     statistics  
   - **IMU features:** statistical features derived from accelerometer and gyroscope
     signals  
   - **Sensor fusion:** feature-level concatenation of EMG and IMU features  

5. **Model training and evaluation**  
   Separate classifiers are trained for:
   - EMG-only features  
   - IMU-only features  
   - Fused EMG-IMU features  

---

## Models

The project evaluates classical machine learning classifiers trained on extracted
features. Model performance is compared across modalities to assess the impact of
sensor fusion.

Evaluation metrics include:
- Classification accuracy
- Confusion matrix analysis

---

## Results

The results demonstrate that combining EMG and IMU information improves gesture
recognition performance compared to unimodal approaches. This supports the hypothesis
that muscle activation and kinematic data provide complementary information for gesture
classification.

An example confusion matrix from the fused EMG–IMU model is shown below:

<img width="1327" height="1181" alt="confusion_matrix" src="https://github.com/user-attachments/assets/299038dd-17af-4bc3-910a-88bac37dd7ab" />

---

## Future Work

Future works include:
- Real-time gesture recognition using streaming EMG–IMU data
- Online inference and latency optimisation for real-time deployment
- Evaluation on a larger and more diverse subject pool
- Integration with robotic or assistive systems for closed-loop control
