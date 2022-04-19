import matplotlib.pyplot as plt
import os
import numpy as np
import thunder as td

current_working_dir = ""

if current_working_dir != "":
    os.chdir(current_working_dir)
else:
    print(f"""\nYour current working directory is {os.getcwd()}. Ensure that the project's folder is in this directory, otherwise change the cwd with 
    os.chdir('dirname') from the shell.""")    


print("\nLoading the data...")
images_ds = td.images.frombinary(os.getcwd()+'/neuroprocessing/data/fish_long/*.bin', order = 'F', engine = sc)     #   this is a distributed dataset
print(f"\nThe image dataset is composed of {images_ds.count()} images each with dimensions {images_ds.first().shape}")																				 

print("\nConvert the image data to time series of voxels...")
series_ds = images_ds.toseries()
print(f"\nThe series dataset is composed of {series_ds.count()} records one for each voxel. A {series_ds.first().shape} long time series corresponds to each voxel. ")

print("\nNormalize the time series data by subtracting and dividing with the 'mean' + 'a small default offset'...")
normalized = series_ds.normalize(method = 'mean')         #    a distributed dataset as well as the series_ds          

print("\nThe distribution of the standard deviation of the greyscale values of each voxel across time is depicted in the hist.png image for a 1000 sample.   ")                
std_nor = normalized.map(lambda n : n.std())
sample = std_nor.sample(1000)
 
fig = plt.figure(figsize = (8,8)) 
plt.hist(sample.values, bins= 20)
plt.savefig('neuroprocessing/hist.png')
plt.clf()

print("""\n From the histogram we observe that the voxels that are 'active' most of the time are those exhibiting standard deviation above 0.1 . 
   The time series of image intensity for 50 of those most 'active' voxels look like as it is depicted in the image filtered.png. Waves many of
   which share the same phase.""")
filtered = normalized.filter(lambda e : e.std() > 0.1)    
plt.plot(filtered.sample(50).values.T)
plt.savefig('neuroprocessing/filtered.png')
plt.clf()

from pyspark.mllib.clustering import KMeans

print("\nLets cluster the normalized voxel measurements using KMeans in order to identify neurons' various patterns of behavior in the brain.... ")
print("\nWe will attempt clustering of the voxels using a varying number of clusters, each time (different k values).")
ks = [5, 10, 15, 20, 30, 50, 100, 200]
models = []
for k in ks:
	models.append( KMeans.train(normalized.values._rdd.values(),k))                       
	
	
def error_1(series, model):                   
	eucl_distances = series.map(lambda v : ((model.clusterCenters[model.predict(v)] -  v).dot(model.clusterCenters[model.predict(v)] -  v))**0.5)
	return   eucl_distances.toarray().sum()
	
def error_2(series, model):
	return  model.computeCost(series.values._rdd.values())

print("""\nThe error in clustering is computed as the sum of the euclidean distances (ED) between the series and the center of 
the cluster to which each series was classified. """)	
print("""\nA second error metric is used, the sum of squared distances (SD) of the series from the centers of the clusters to 
which the series where classified.
  The curves of error values estimated using both error metrics for the various KMeans models are depicted in the error_k.png image. """)	
er1_est = []
er2_est	= []
for model in models:
	er1_est.append(error_1(normalized,model))
	er2_est.append(error_2(normalized,model))

plt.plot(ks, er1_est/sum(er1_est), label = "error - ED")
plt.plot(ks, np.asarray(er2_est)/sum(er2_est), label = "error - SD")
plt.xlabel("k")
plt.ylabel('Error estimate')
plt.legend()
plt.savefig("neuroprocessing/error_k.png")
plt.clf()
	
print("\nThe cluster centers for k = 10 appear in the image centers_of_10_clusters.png.")    
model10 = models[1]
model100 = models[6]
centers100 = model100.clusterCenters
centers10 = model10.clusterCenters
plt.plot(np.asarray(centers10).T)
plt.savefig('neuroprocessing/centers_of_10_clusters.png')
plt.clf()

print("\n Model selection....")

print("""\nBy the error metric diagrams' retrospection the model with k=100 will be used for the clustering, 
where the error has its lowest value.""")

print("""\n Lets view an image with the value of each voxel encoded by the number of its representative cluster. The image depicts
functional groups that are estimated to have been formed by voxels that show similar activity most of the time. The image is 
shown in enc_img.tif.""")
encoded = normalized.map(lambda v : model100.predict(v)).toarray()

plt.imshow(encoded[:, :, 0], interpolation='nearest',aspect='equal', cmap='gray')
plt.savefig('neuroprocessing/enc_img.tif')
plt.clf()

