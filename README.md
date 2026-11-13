# Smart Solar Thermal Monitoring and Predictive Control System with Autonomous Fault Mitigation

## Project Overview

The Smart Solar Thermal Monitoring and Control System (STMaCS) is an IoT-enabled research prototype developed to investigate intelligent monitoring, predictive control and fault mitigation in solar thermal energy systems.

The project explores the integration of embedded systems, wireless sensor networks, cloud-based data acquisition, machine learning, thermal modelling and control strategies to improve the monitoring and operational reliability of solar thermal installations.

Developed around a 300-litre HOTSET solar thermal system at the University of Botswana, the project combines physical sensor integration with software-based simulation, historical data analysis and digital-twin development.

The system provides a framework for monitoring thermal behaviour, analysing historical performance, forecasting future operating conditions and investigating proactive control and fault-response strategies.

## Research Context and Funding

This research was undertaken at the **University of Botswana** with support from the **SOLTRAIN+ programme**, funded by the **AEE INTEC**.

The project contributes to research into intelligent renewable energy systems, particularly the application of IoT technologies, predictive analytics and embedded control in solar thermal energy management.

## Key Research Contributions and System Features

### 1. Embedded Systems and Real-Time Monitoring

- Integration of ESP32-based embedded hardware for sensor acquisition.
- Temperature monitoring using digital temperature sensors.
- Integration and testing of flow-sensing functionality.
- Acquisition and transmission of sensor measurements for remote monitoring.
- Development of an IoT communication architecture connecting embedded hardware to cloud infrastructure.

### 2. Cloud Connectivity and Data Management

- Integration with Firebase Realtime Database for telemetry storage and communication.
- Development of data structures for transmitting and retrieving system measurements and operational states.
- Integration of embedded hardware data with a Python-based monitoring platform.

### 3. Thermal Digital Twin Development

- Development of a software-based representation of the solar thermal system.
- Modelling of tank thermal behaviour, heat transfer and operating conditions.
- Simulation of temperature changes and system responses under different operating scenarios.
- Integration of digital-twin visualisation into the monitoring dashboard.

### 4. Historical Data Analysis and Predictive Modelling

- Analysis and preprocessing of historical solar thermal system datasets.
- Investigation of missing measurements, irregular sampling intervals and invalid sensor values.
- Development and evaluation of machine-learning models for thermal forecasting.
- Comparison of forecasting approaches across different prediction horizons.
- Investigation of how predictive information can support proactive thermal management.

### 5. Predictive Control

The project investigates predictive control strategies for maintaining hot-water availability while minimising unnecessary auxiliary heating.

The control approach considers predicted thermal conditions and expected hot-water demand to inform heating decisions.

Predictive control strategies were explored through simulation and digital-twin-based evaluation rather than being presented as fully validated control of the physical installation.

### 6. Fault Detection and Hardware-in-the-Loop Testing

- Development of fault scenarios involving abnormal flow conditions and thermal operating states.
- Investigation of fault detection and mitigation strategies.
- Hardware-in-the-loop testing using representative actuator hardware.
- Integration of fault-event information into the monitoring and dashboard architecture.

Hardware-in-the-loop testing provided a means of investigating system responses without requiring every mitigation action to be implemented on the full solar thermal installation.

### 7. Web-Based Monitoring Dashboard

Developed using **Python and Streamlit**, the dashboard brings together system monitoring, visualisation and analysis.

The platform includes:

- Digital twin visualisation.
- Live sensor monitoring.
- Historical data visualisation.
- Fault-event logging.
- System status indicators.
- Configuration and settings interfaces.

The dashboard integrates embedded-system telemetry with software-based simulation and analysis.

## System Architecture

```text
       SOLAR THERMAL SYSTEM
                |
                |
       Embedded Sensor Network
                |
                |
          ESP32 Hardware
                |
                |
       Cloud Data Infrastructure
          (Firebase RTDB)
                |
                |
       Python / Streamlit Platform
                |
       +--------+---------+
       |        |         |
       |        |         |
   Monitoring  Digital   Historical
   Dashboard    Twin     Data Analysis
       |        |         |
       +--------+---------+
                |
                |
       Predictive Modelling
                |
                |
         Predictive Control
                |
                |
       Fault Detection and
       Mitigation Investigation
                |
                |
       Hardware-in-the-Loop
             Testing
```

## Technologies and Tools

### Programming and Software
- Python
- C/C++
- Streamlit
- MATLAB/Simulink
- Data analysis and visualisation

### Embedded Systems and IoT
- ESP32 microcontrollers
- Digital temperature sensors
- Flow-sensing hardware
- Firebase Realtime Database
- Wireless communication and telemetry

### Machine Learning and Modelling
- Historical time-series data analysis
- Machine-learning-based forecasting
- Predictive modelling
- Thermal system simulation
- Digital-twin development

### Control and Energy Systems
- Solar thermal energy systems
- Thermal monitoring
- Predictive control
- Fault detection and mitigation
- Hardware-in-the-loop testing

## Research Scope and Limitations

The project combines physical hardware experimentation with simulation and software-based evaluation.

Not all proposed functions were validated through full-scale operation on the physical solar thermal installation. In particular:

- The digital twin was developed as a software-based model and was not fully calibrated against the physical installation.
- Predictive control was evaluated through modelling and simulation rather than complete physical closed-loop deployment.
- Hardware-in-the-loop tests used representative actuator hardware rather than a fully integrated installation.
- Some communication and auxiliary notification functions remain subject to further development and validation.

These limitations define opportunities for future research and further system development.

## Research Output

The project formed part of research presented at the **International Sustainable Energy Conference (ISEC 2026)** in Graz, Austria, on 15 April 2026.

The work contributed to research into intelligent monitoring and predictive control for solar thermal energy systems.

ISEC Proceeding: Design and Implementation of a Smart Solar Thermal Monitoring and Predictive Control System With Autonomous Fault Mitigation 
DOI: https://doi.org/10.52825/isec.v2i.3361

## Author

**Gorata Matsapa**  
Electronic Engineering Graduate  
University of Botswana

Research interests:
- Embedded Systems and IoT
- Intelligent Energy Systems
- Machine Learning for Engineering Applications
- Predictive Control
- Renewable Energy Technologies
- Digital Twins and Cyber-Physical Systems

## Project Status

The research prototype and associated software, modelling and experimental work have been developed and documented. Further work may extend physical system integration, model calibration, closed-loop control validation and autonomous fault mitigation.
