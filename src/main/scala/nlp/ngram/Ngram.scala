//Author: Stephen Pryor
//Feb 26, 2013

package jarvis.nlp

/**
 * A class for creating ngram language models.
 *
 * @param n is the type of ngram you wish to create (i.e. n=2 creates a bigram, n=3 creates a trigram, etc.)
 * @param sentences is and indexed sequence of tokenized sentences to use as training data
 * @param alpha is the value to use for laplace alpha smoothing
 * @param useBackoff tells the model whether or not to use simple ("stupid") backoff or not
 */
class Ngram(n:Int, 
            sentences: IndexedSeq[IndexedSeq[String]], 
            alpha: Double = 0.5, 
            useBackoff: Boolean = true) {
  import collection.mutable.HashMap  
  
  private[this] val bookends = (1 to n-1).map(_ => "<s>").toIndexedSeq
  private[this] val ngramCounts = (1 to n)
                    .map(i => countNgrams(i, sentences))
                    .toIndexedSeq
  private[this] val totalUnigrams = ngramCounts(0).values.sum.toDouble
  
  /**
   * Returns the log probability of a tokenized sentence
   *
   * @param sentence is a tokenized sentence
   * @return A double of the log probability of the sentence
   */
  def apply(sentence: IndexedSeq[String]) = {
    prob(sentence)
  }
  
  /**
   * Returns the log probability of a tokenized sentence
   *
   * @param sentence is a tokenized sentence
   * @return A double of the log probability of the sentence
   */
  def prob(sentence: IndexedSeq[String]) = {
    extractNgrams(n, bookends ++ sentence)
    .map(ngram => math.log(probNgram(ngram)))
    .sum
  }
  
  /**
   * Returns the perplexity score of the model on a given dataset
   *
   * @param sentences is a indexed sequence of tokenized sentences
   * @return A double of the perplexity score of the model for the given model
   */
  def computePerplexity(sentences: IndexedSeq[IndexedSeq[String]]) = {
    var numtokens = 0
    var totalProb = 0.0
    sentences.foreach(sentence => {
      numtokens += sentence.length
      totalProb += prob(sentence)
    })
    math.pow(2, -totalProb/numtokens)  
  }
  
  /**
   * Returns ngrams of a tokenized sequence
   *
   * @param n the value for the size of ngram
   * @param sentences is a indexed sequence of tokenized sentences
   * @return An indexed sequence of ngrams
   */
  private[this] def extractNgrams(n: Int, sentence: IndexedSeq[String]) = {
    sentence.sliding(n).map(_.mkString(" ")).toIndexedSeq
  }
  
  /**
   * Returns the probability of a given ngram
   *
   * @param ngram the ngram to retrieve the probability for
   * @param n the value for the type of ngram provided, defaults to the class n but can be specified for backoff
   * @return A double for the probability of the ngram
   */
  private[this] def probNgram(ngram: String, n:Int = n - 1):Double = {
    if(n <= 0) {
      (ngramCounts(0)(ngram) + alpha) / (totalUnigrams + alpha*ngramCounts(0).size.toDouble)
    } else {
      val prevGram = ngram.replaceAll("\\s+[^\\s]+$", "")
      val prevCount = ngramCounts(n-1)(prevGram)
      if(prevCount < 1.0 && useBackoff) { //if prevGram does not exist, backoff
        probNgram(ngram.replaceAll("^[^\\s]+\\s+", ""), n-1)
      } else {
        val currCount = ngramCounts(n)(ngram)
        (currCount + alpha) / (prevCount + alpha*ngramCounts(n-1).size.toDouble)
      }
    }
  }
  
  /**
   * Returns a map of ngrams mapped to their counts in the dataset
   *
   * @param n the value for the size of ngram
   * @param sentences is a indexed sequence of tokenized sentences
   * @return A map of ngrams mapped to their counts
   */
  private[this] def countNgrams(n:Int, sentences: IndexedSeq[IndexedSeq[String]]) = {
    val counts = HashMap[String,Double]().withDefaultValue(0.0)
    sentences.foreach(sentence => {
         extractNgrams(n, bookends ++ sentence)
         .foreach(gram => counts(gram) += 1)
    })
    counts
  }
}

