//Author: Stephen Pryor
//Feb 24, 2013

/**
 * Note: This code is essentially a modified version of 
 * Dr. Jason Baldridge's Kmeans implementation in nak.
 * https://github.com/jasonbaldridge/nak
 *
 * I have made only slight modifications such as adding
 * Kmeans++ as an alternative to random centroid 
 * initialization and some minor code restructuring.
 **/


package jarvis.cluster

import jarvis.math._

/**
 * A class for running the Kmeans algorithm 
 *
 * @param points an indexed sequence of points to be clustered
 * @param distance the DistanceFunction to use to compute distance between points
 * @param maxChangeInDispersion the change in dispersion to tell the algorithm to terminate
 * @param maxIter is the maximum number of iterations to run k-means for
 * @param useRandomSeed tells the algorithm whether or not to use a random seed or a non-random number (for repeatable results)
 * @param useRandomInitialization tell the algorithm to use Kmeans++ initialization (the default) or random initialization
 * @param defaultRandomSeed the default seed value to use if useRandomSeed is false
 */
class Kmeans(
      points: IndexedSeq[Point],
      distFunc: DistanceFunction = DistanceFunction("e"),
      minChangeInDispersion: Double = 0.0001,
      maxIter: Int = 200,
      useRandomSeed: Boolean = true,
      useRandomInitialization: Boolean = false,
      defaultRandomSeed: Int = 13
      ){

  import scala.util.Random

  lazy private val random = if(useRandomSeed) new Random()
                            else new Random(defaultRandomSeed)
  
  def run(k: Int, attempts: Int = 20): (Double,  IndexedSeq[Point]) = {
    (1 to attempts).map{ _ => 
      moveCentroids(chooseCentroids(k))
    }.minBy(_._1)
  }

  private[this] def chooseCentroids(k: Int) = if(!useRandomInitialization) chooseKmeansPlusPlusCentroids(k)
                                              else chooseRandomCentroids(k)

  private[this] def moveCentroids(centroids: IndexedSeq[Point]): (Double,  IndexedSeq[Point]) = {
    def step(centroids: IndexedSeq[Point], 
             prevDispersion: Double, 
             iter: Int): (Double, IndexedSeq[Point]) = {
      if(iter > maxIter) {
        (prevDispersion, centroids)
      } else {
        val (dispersion, assignments) = assignClusters(centroids)

        if((prevDispersion - dispersion) < minChangeInDispersion)
          (prevDispersion, centroids)
        else
          step(updateCentroids(assignments), dispersion, iter + 1)
      }
    }
    step(centroids, Double.PositiveInfinity, 1)
  }

  private[this] def assignClusters(centroids: IndexedSeq[Point]) = {
    val (squaredDistances, assignments) = points.map(assignPoint(_,centroids)).unzip
    (squaredDistances.sum, assignments)
  }

  private[this] def assignPoint(point: Point, centroids: IndexedSeq[Point]) = {
    val (shortestDist, assignment) = centroids
      .map(distFunc(_,point))
      .zipWithIndex
      .min
    (shortestDist * shortestDist, assignment)
  }

  private[this] def updateCentroids(assignments: IndexedSeq[Int]) = {
    assignments
      .zip(points)
      .groupBy(k => k._1) //group by cluster assignment
      .mapValues(cluster => cluster.map(_._2).reduce(_ + _) / cluster.length.toDouble)
      .map(_._2)
      .toIndexedSeq
  }

  private[this] def chooseRandomCentroids(k: Int) = {
    random.shuffle(points).take(k)
  }

  private[this] def chooseKmeansPlusPlusCentroids(k: Int) = {

    def stepKmeansPlusPlus(centroids: IndexedSeq[Point], k: Int): IndexedSeq[Point] = {
      if(k < 1) {
        centroids
      } else {
        val (shortestDistances, _) = points.map(assignPoint(_, centroids)).unzip
        val denominator = shortestDistances.sum
        var i = 0
        while(random.nextDouble() < shortestDistances(i)/denominator) {
          i = if(i >= shortestDistances.length) 0
              else i + 1
        }
        stepKmeansPlusPlus(centroids ++ IndexedSeq(points(i)), k-1)
      }
    }

    stepKmeansPlusPlus(random.shuffle(points).take(1), k)
  }
}
