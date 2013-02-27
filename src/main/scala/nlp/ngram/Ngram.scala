//Author: Stephen Pryor
//Feb 26, 2013

package jarvis.nlp

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
  
  def apply(sentence: IndexedSeq[String]) = {
    prob(sentence)
  }
  
  def prob(sentence: IndexedSeq[String]) = {
    extractNgrams(n, bookends ++ sentence)
    .map(ngram => math.log(probNgram(ngram)))
    .sum
  }
  
  def computePerplexity(sentences: IndexedSeq[IndexedSeq[String]]) = {
    var numtokens = 0
    var totalProb = 0.0
    sentences.foreach(sentence => {
      numtokens += sentence.length
      totalProb += prob(sentence)
    })
    math.pow(2, -totalProb/numtokens)  
  }
  
  private[this] def extractNgrams(n: Int, sentence: IndexedSeq[String]) = {
    sentence.sliding(n).map(_.mkString(" ")).toIndexedSeq
  }
  
  private[this] def probNgram(ngram: String, n:Int = n - 1):Double = {
    if(n <= 0) {
      (ngramCounts(0)(ngram) + alpha) / (totalUnigrams + alpha*ngramCounts(0).size.toDouble)
    } else {
      val prevGram = ngram.replaceAll("\\s+[^\\s]+$", "")
      val prevCount = ngramCounts(n-1)(prevGram)
      if(prevCount < 1.0 && useBackoff) { //if prevGram does exist, backoff
        probNgram(ngram.replaceAll("^[^\\s]+\\s+", ""), n-1)
      } else {
        val currCount = ngramCounts(n)(ngram)
        (currCount + alpha) / (prevCount + alpha*ngramCounts(n-1).size.toDouble)
      }
    }
  }
  
  private[this] def countNgrams(n:Int, sentences: IndexedSeq[IndexedSeq[String]]) = {
    val counts = HashMap[String,Double]().withDefaultValue(0.0)
    sentences.foreach(sentence => {
         extractNgrams(n, bookends ++ sentence)
         .foreach(gram => counts(gram) += 1)
    })
    counts
  }
}

