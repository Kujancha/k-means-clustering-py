import math
import numpy as np
import random


# this is the function for finding the Eucledian Distance between two ppints lol
def eucledian(p1, p2):
    if np.size(p1) != np.size(p2):
        print(f"Error, the dimensions of the two points does not match")
        return
        
    return math.sqrt(sum((a-b) ** 2 for a,b in zip(p1, p2)))  


class Kmeans:
    def __init__(self, k = 3, max_ite = 100):
        self.k = k
        self.max_ite = max_ite
        self.centroids = []
     
     
     # the data should be a list of list of integers (or float)    
    def fit(self , data:list): 
        self.centroids = random.sample(data, self.k)  # this assigns a list of k random elements to centroids (our initial guess)

        for _ in range(self.max_ite):
            clusters = [[] for _ in range(self.k)]
              #the above creates a list of list amounting to the numeric value of K i.e to store each cluster
              
            for point in data:
                distances = [eucledian(point, centroid) for centroid in self.centroids]
                cluster_index = distances.index(min(distances))
                clusters[cluster_index].append(point)
                
            # to calculate new centroids
            
            new_centroids = []
            
            for cluster in clusters:
                if cluster:
                    mean = [sum(dim)/ len(cluster) for dim in zip(*cluster)]
                    new_centroids.append(mean)
                else:
                    new_centroids.append(random.choice(data))
                    
            if new_centroids == self.centroids:
                break
            
            self.centroids = new_centroids
    
    
    def predict(self,point):
        distances = [eucledian(point, centroid) for centroid in self.centroids]
        return distances.index(min(distances))
    
    


def main():
    data = [
    [1, 2], [2, 3], [3, 1],   # Cluster 1
    [8, 8], [9, 10], [10, 9], # Cluster 2
    [25, 30], [24, 27], [26, 29]  # Cluster 3
]

    model = Kmeans(k = 3)
    model.fit(data)

    for point in data:
        print(f"Point {point} → Cluster {model.predict(point)}")

    print("Centroids:", model.centroids)




if __name__ == "__main__":
    main()
