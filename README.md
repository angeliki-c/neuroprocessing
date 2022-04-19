## neuroprocessing

### Functional Analysis of Neurons in the Brain from Brain Images 
    
	
	
Techniques followed

	K-Means clustering on normalized data coming from images of the brain of zebrafish taken at specific time
  	frames. The 'thunder' python library [1], developed for facilitating large scale vector image and time 
	series analysis, is used for processing many images of high resolution across time, with the aim of gaining
	insight at the functional operation of groups of neurons or of single neurons in the zebrafish brain. This
	library originally designed in specific for the analysis of neuroimaging data is successfull in decoupling 
	the storage management of the data from the underlying domain model of the application, by taking advantage 
	of the capabilities that the Spark framework offers in distributed data management. Despite the library was 
	designed with processing image and video data as main area of application in mind, it has the capability to
	be used in the analysis of multimodal data (data coming from different fields).
	

  
Data set

 	In this study, 243 3D images from zebrafish's brain is analyzed [2]  
	


Challenges

  	The analysis of the dynamics observed in brain's function involves transformations of the data between the
	two domains image and time domain.
	The transformation of the data between the two domains of reference, space and time in this case study, is 
	an expensive operation, which's cost the 'thunder' library tries to diminish when it comes in.
	

 
Training process

  	The clustering is performed using the KMeans algorithm built in pyspark mllib. The training with KMeans
	involves a heuristic for the selection of the initial centers of the clusters, called 'k-means||'. The
	number of iterations is set to 100, unless the algorithm converges in the computation of the cluster 
	centers, as close as less than 0.0001.
  	6 models have been trained each one corresponding to a different value for k. The selection of the model,
	for the clustering, is based on the minimization of two error metrics, the sum of the euclidean distances 
	of the data points from the centers of the clusters to which the data points have been classified and the
	sum of the squared distance of the data points from the centers of the clusters, correspondingly.
	
	

Evaluation

  	For the evaluation of the clustering the sum of the squared distances of the data points from the cluster
	centers is used, though there are other metrics, such as the Shilouette metric, which may give as an even
	more reliable view on the quality of the clustering. 

  	Evaluation on whether the voxel groups identified by the algorithm, really correpond to potential functional
	groups of neurons in the zebrafish brain has not been tried in this study.
	


Code

   	neuroprocessing.py
   
   	All can be run interactively with pyspark shell or by submitting e.g. 
	exec(open("project/location/neuroprocessing/neuroprocessing.py").read()) for an all at once execution. The 
	code has been tested on a Spark standalone cluster. For the Spark setting, spark-3.1.2-bin-hadoop2.7 bundle
	has been used.
   	The external python packages that are used in this implementation exist in the requirements.txt file. Install
	with: 
	   	pip install -r project/location/neuroprocessing/requirements.txt
   	This use case is inspired from the series of experiments presented in [3], though it deviates from it, in the
   	programming language, the setting used and in the analysis followed.

References  

  	1. http://docs.thunder-project.org/image-loading
	2. https://github.com/sryza/aas/tree/master/ch11-neuro/fish-long
	3. Advanced Analytics with Spark, Sandy Ryza, Uri Laserson, Sean Owen, & Josh Wills
	
