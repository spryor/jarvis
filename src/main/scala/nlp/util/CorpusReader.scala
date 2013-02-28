//Author: Stephen Pryor
//Feb 27, 2013

package jarvis.nlp.util

import java.io.File

/* -------------------- Tokenizer - START -------------------- */

trait Tokenizer {
  def apply(text: String): IndexedSeq[String] 
}

object SimpleTokenizer extends Tokenizer {
  lazy val tokenPattern = """([<]?[a-zA-Z]+(['][a-zA-Z]+)?[>]?)""".r
    
  override def apply(text: String): IndexedSeq[String] = {
    (tokenPattern findAllIn text).toIndexedSeq
  }
}

/* -------------------- Tokenizer - END -------------------- */

/* -------------------- CorpusParser - START -------------------- */

trait FileParser{
  def parse(text: String, tokenize: Tokenizer): IndexedSeq[IndexedSeq[String]]
}

trait PENNParser extends FileParser {
  val pattern = """([^\s]+/[^\s]+)""".r
  
  override def parse(text: String, tokenize: Tokenizer): IndexedSeq[IndexedSeq[String]] = text
            .replaceAll("[*x]{2,}.*[*x]", "") // Remove any comments
            .replaceAll("(=+)(\\s*=+)*", "$1")
            .trim
            .split("=+")
            .filter(_.length > 0)
            .flatMap(x => {
              (pattern findAllIn x.trim)
              .mkString(" ")
              .split("/\\.")
              .map(_.replaceAll("""/[^\s]+|[\\]""", "")
                    .replaceAll("\\s+([n]?'[^\\s]+)|([;:])\\s*[;:]", "$1")
                    .replaceAll("[\\d]+", "<num>")
                    .trim)
            }).filter(_.length > 1)
            .map(tokenize(_))
            .toIndexedSeq
}

/* -------------------- CorpusParser - END -------------------- */

/* -------------------- CorpusReader - START -------------------- */

abstract class CorpusReader extends FileParser {
  def apply(directory:String, tokenizer: Tokenizer = SimpleTokenizer) = {
    val dataset = new File(directory)
    if(!dataset.isDirectory) {
      println("Error[CorpusReader]: "+directory+" is not a directory.")
      System.exit(0)
    }
    grabFiles(dataset).flatMap(file => {
      parse(readPOSFile(file), tokenizer)
    }).filter(_.length > 1)
  }
  
  def readPOSFile(path:File) = {
    io.Source.fromFile(path).getLines.mkString("\n")
  }
  
  def grabFiles(f: File): IndexedSeq[File] = {
    val files = f.listFiles
    files.filter(file => !file.isDirectory && file.toString.endsWith(".pos")) ++ files.filter(_.isDirectory).flatMap(grabFiles)
  }
}

/* -------------------- CorpusReader - END -------------------- */

//Creates an object to read read .pos files from the Penn treebank
object PENNReader extends CorpusReader with PENNParser


