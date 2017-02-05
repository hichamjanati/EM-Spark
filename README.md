# READ ME 

Gaussian mixture models EM Algorithm with pySpark 

Authors: Johann Faouzi - Hicham Janati 

----------------------------------------------------------


## 1- Code: 

### File generate.py: 

Contains the function EM_sample used to generate Gaussian mixtures. 

### File EM.py: 

   Contains the two objects:

- Class EM_Spark: EM Algorithm written in a spark mapreduce style.

- Class EM_noSpark: Classical EM Algorithm with numpy  


## 2- Notebooks:

  - Test.ipynb Simulation examples.

  - Time evaluation of the two objects fitting methods on simulated datasets. 
