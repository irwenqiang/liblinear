package tw.edu.ntu.csie.liblinear

class Feature(val index: Int, val value: Double) extends Serializable {}

/**
 * DataPoint represents a sparse data point with label.
 *
 * @param x features represented in an Array of Feature
 * @param y label
 */
class DataPoint(val x : Array[Feature], val y : Double) extends Serializable
{

	def getMaxIndex() : Int = 
	{
		this.x.last.index
	}

	def genTrainingPoint(n : Int, b : Double, posLabel : Double) : DataPoint = 
	{
		var x : Array[Feature] = null
		var y = if(this.y == posLabel) 1.0 else -1.0
		if(b < 0)
		{
			x = this.x
		}
		else
		{
			x = new Array[Feature](this.x.size + 1)
			this.x.copyToArray(x, 0)
			x(x.size-1) = new Feature(n-1, b)
		}
		new DataPoint(x, y)
	}
}
