//Author: Stephen Pryor
//Feb 25, 2013

package jarvis.classify

/**
 * A class for creating a simple naive bayes classifier.
 * In this document, a feature vector refers to a sequence of features and their associated counts in a document.
 *
 * @param featureVectors is a map of labels to document feature vectors. Each key in the map is a class label. Each feature vector is an indexed sequence of a feature to the count of that feature in a particular document
 * @param alpha is the smoothing parameter for laplace alpha smoothing. 0.0 is no smoothing.
 */
class NaiveBayesClassifier[LabelType, FeatureType](
      featureVectors: Map[LabelType, IndexedSeq[IndexedSeq[(FeatureType, Int)]]],
      alpha: Double){

  private val featureCounts = computeFeatureCounts()
  private val numDocumentsPerLabel = featureVectors.mapValues(_.length.toDouble)
  private val totalDocuments = numDocumentsPerLabel.values.sum.toDouble
  private val numWordsPerLabel = featureCounts.mapValues(_.values.sum.toDouble)
 
  /**
   * Takes a document feature vector and returns the predicted label. 
   *
   * @param featureVector is the feature vector for a document to be classified
   * @return A label of type LabelType
   */
  def apply(featureVector: IndexedSeq[(FeatureType, Int)]): LabelType = {
    featureCounts.keys
      .map(label => {
        val denominator = numWordsPerLabel(label) + alpha*featureCounts(label).size.toDouble
        (probOfLabel(label) + featureVector.map{ case(feature, count) => {
          count.toDouble*math.log((featureCounts(label)(feature) + alpha)/denominator)          
        }}.sum, label)
      }).min(Ordering.by[(Double, LabelType),Double](_._1))._2
  }

  /**
   * Returns the probablity of a class label.
   *
   * @param label is the label to retrieve the probablity for.
   * @return A double representing the probability of label
   */
  private[this] def probOfLabel(label: LabelType) = {
    math.log(numDocumentsPerLabel(label)/totalDocuments)
  }

  /**
   * Converts the featureVectors into counts of features per label
   *
   * @return A map from labels to features and from features to counts
   */
  private[this] def computeFeatureCounts(): Map[LabelType, Map[FeatureType, Double]] = {
    featureVectors.mapValues(_.flatten
      .groupBy(_._1)
      .mapValues(_.map(_._2)
                  .sum
                  .toDouble) 
      .withDefaultValue(0.0))
  }
} 
 
