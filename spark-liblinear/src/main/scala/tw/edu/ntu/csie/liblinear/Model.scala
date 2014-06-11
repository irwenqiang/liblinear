package tw.edu.ntu.csie.liblinear

import org.apache.spark.util.Vector
import java.io.FileOutputStream
import java.io.FileInputStream
import java.io.ObjectOutputStream
import java.io.ObjectInputStream
import scala.Predef
import tw.edu.ntu.csie.liblinear.SolverType._

/** 
 * A linear model stores weights and other information.
 *
 *@param param user-specified parameters
 *@param w the weights of features
 *@param nrClass the number of classes
 *@param bias the value of user-specified bias
 */
class Model(val param : Parameter, labelSet : Array[Double]) extends Serializable
{
	var label : Array[Double] = labelSet.sortWith(_ < _)
	val nrClass : Int = label.size
	// weight is a m * f matrix, with row a weight vector for each class
	// m: the number of classes
	// f: the number of features
	var w : Array[Vector] = new Array(nrClass)
	var bias : Double = -1.0

	def setBias(b : Double) : this.type = 
	{
		this.bias = b
		this
	}

	def predictValues(features : Array[Feature]) : Array[Double] = 
	{
		var values = Array.fill(nrClass)(0.0)
		val lastIndex = w(0).length - 1
		if(nrClass == 2)
		{
			for(feature <- features)
			{
				if(feature.index <= lastIndex)
				{
					values(0) += feature.value * w(0)(feature.index)
				}
			}
			if(bias >= 0)
			{
				values(0) += bias*w(0)(lastIndex)
			}
		}
		else
		{
			for(i <- 0 until nrClass) 
			{
				for(feature <- features) 
				{
					if(feature.index <= lastIndex) 
					{
						values(i) += feature.value * w(i)(feature.index)
					}
				}
				if(bias >= 0) 
				{
					values(i) += bias*w(i)(lastIndex)
				}
			}
		}
		values
	}

	/** 
	 *Predict a label given a DataPoint.
	 *@param point a DataPoint
	 *@return a label
	 */
	def predict(point : DataPoint) : Double = 
	{
		val values = predictValues(point.x)
		var labelIndex = 0
		if(nrClass == 2) 
		{
			if(values(0) < 0) 
			{
				labelIndex = 1
			}
		}
		else
		{
			for(i <- 1 until nrClass)
			{
				if(values(i) > values(labelIndex))
				{
					labelIndex = i
				}
			}
		}
		label(labelIndex)
	}

	/** 
	 *Predict probabilities given a DataPoint.
	 *@param point a DataPoint
	 *@return probabilities which follow the order of label
	 */
	def predictProbability(point : DataPoint) : Array[Double] =
	{
		Predef.require(param.solverType == L2_LR, "predictProbability only supports for logistic regression.")
		var probEstimates = predictValues(point.x)
		probEstimates = probEstimates.map(value => 1.0/(1.0+Math.exp(-value)))
		if(nrClass == 2)
		{
			probEstimates(1) = 1.0 - probEstimates(0)
		}
		else
		{
			var sum = probEstimates.sum
			probEstimates = probEstimates.map(value => value/sum)
		}
		probEstimates
	}

	/** 
	 * Save Model to the local file system.
	 *
	 * @param fileName path to the output file
	 */
	def saveModel(fileName : String) =
	{
		val fos = new FileOutputStream(fileName)
		val oos = new ObjectOutputStream(fos)
		oos.writeObject(this)
		oos.close
	}
}

object Model
{

	/** 
	 * load Model from the local file system.
	 *
	 * @param fileName path to the input file
	 */
	def loadModel(fileName : String) : Model =
	{
		val fis = new FileInputStream(fileName)
		val ois = new ObjectInputStream(fis)
		val model : Model = ois.readObject.asInstanceOf[Model]
		ois.close
		model
	}
}

	
