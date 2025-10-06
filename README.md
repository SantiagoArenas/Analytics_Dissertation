# Analytics_Dissertation
A repository used for my final dissertation project about analytical insurance policies. 

## Data sources
Because of a lack of first-party propietary data, this is how I plan to get all of the data necesary for this project. Using a combination of these publicly availeable Data, one might be able to acquire the neccesary information to complete this project.
### Vehicle Telematics
Information about how cars are driven.
1. [**SHRP2 — Naturalistic Driving Study (NDS)**](https://insight.shrp2nds.us/data/index)
I have requested information to them via email, seeing as one cannot access the data without getting permission.
2. [**highD dataset**](https://levelxdata.com/highd-dataset/)
I have gotten access to this data. Extensive and amazing.
3. [**Driving Behavior Dataset**](https://data.mendeley.com/datasets/jj3tw8kj6h/3)
Dataset for modeling risky driver behaviors based on accelerometer (X,Y,Z axis in meters per second squared (m/s2)) and gyroscope (X,Y, Z axis in degrees per second (°/s) ) data. Very few drivers (3), small scale, no policy/claim labels, no exposure metadata.
### Insurance Policies
Information about insurance policies.
1. [**OpenICPSR — “Dataset of an actual motor vehicle insurance portfolio”**](data/insurance/openICPSR/)
Found the dataset, incluiding a description of the variables, the whole dataset, and a sample type claim.
1. [**freMTPL2**](data/insurance/freMTPL2/)
Policy data, claims, frequency and severity in France. Accessed and cited if needed.
1. **CA DMV Crash / claims registries & traffic data**
Yet to find this one.
2. [**NCID Private Motor Insurance Data – Ireland**](https://data.gov.ie/en_GB/dataset/ncid-part-2-claims-ultimate-claims)
Ultimate Claims grouped by accident year; aggregated info per accident year. No telematics, no high temporal resolution.
## Auxiliary Data
Useful context to add to the dataset. I plan to connect via API and help get more context into the incidents. Less time consuming and therefore less urgent.
1. **Road Network & Posted Speeds**
2. **Weather information**

