//Author: Stephen Pryor
//Feb 25, 2013

package jarvis.classify

class NaiveBayesClassifier[LabelType, FeatureType](
      featureVectors: Map[LabelType, IndexedSeq[IndexedSeq[(FeatureType, Int)]]],
      alpha: Double){

  private val featureCounts = computeFeatureCounts()
  private val numDocumentsPerLabel = featureVectors.mapValues(_.length.toDouble)
  private val totalDocuments = numDocumentsPerLabel.values.sum.toDouble
  private val numWordsPerLabel = featureCounts.mapValues(_.values.sum.toDouble)
 
  def apply(featureVector: IndexedSeq[(FeatureType, Int)]): LabelType = {
    featureCounts.keys
      .map(label => {
        val denominator = numWordsPerLabel(label) + alpha*featureCounts(label).size.toDouble
        (probOfLabel(label) + featureVector.map{ case(feature, count) => {
          count.toDouble*math.log((featureCounts(label)(feature) + alpha)/denominator)          
        }}.sum, label)
      }).min(Ordering.by[(Double, LabelType),Double](_._1))._2
  }

  private[this] def probOfLabel(label: LabelType) = {
    math.log(numDocumentsPerLabel(label)/totalDocuments)
  }

  private[this] def computeFeatureCounts(): Map[LabelType, Map[FeatureType, Double]] = {
    featureVectors.mapValues(_.flatten
      .groupBy(_._1)
      .mapValues(_.map(_._2)
                  .sum
                  .toDouble) 
      .withDefaultValue(0.0))
  }
} 
 
