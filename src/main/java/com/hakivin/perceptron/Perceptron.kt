package com.hakivin.perceptron

fun main() {
    val w1 = 0.0
    val w2 = 0.0
    val b = 0.0
    val learningRate = 1.0
    val threshold = 0.2
    val epoch = 2
    val data = arrayOf(
            arrayOf(1,1,1,1),
            arrayOf(1,0,1,-1),
            arrayOf(0,1,1,-1),
            arrayOf(0,0,1,-1))
    val result = train(data, w1, w2, b, learningRate, threshold, epoch)
    println(execute(0,0, result, threshold))
}

fun execute(input1:Int, input2:Int, data:Array<Double>, threshold: Double):Int{
    return when {
        input1*data[0] + input2*data[1] + data[2] > threshold -> 1
        input1*data[0] + input2*data[1] + data[2] < threshold -> -1
        else -> 0
    }
}

fun train(data : Array<Array<Int>>, weight1:Double, weight2:Double, bias:Double,learningRate:Double, threshold:Double, epoch:Int):Array<Double>{
    var w1 = weight1
    var w2 = weight2
    var b = bias
    var flag = false
    val array = IntArray(data.size)
    var iter = 0
    while (!flag || iter <= epoch) {
        for (i in data.indices) {
            val arr = data[i]
            val net = (arr[0] * w1) + (arr[1] * w2) + b
            var fnet: Int
            fnet = when {
                net > threshold -> 1
                net < threshold -> -1
                else -> 0
            }
            val wtempt1 = w1
            val wtempt2 = w2
            val btempt = b
            if (fnet != arr[3]) {
                w1 += learningRate * arr[3] * arr[0]
                w2 += learningRate * arr[3] * arr[1]
                b += learningRate * arr[3]
            }
            val deltaw1 = w1 - wtempt1
            val deltaw2 = w2 - wtempt2
            val deltab = b - btempt
            if (deltab == 0.0 && deltaw1 == 0.0 && deltaw2 == 0.0){
                array[i] = 1
            }
            println("net = $net, fnet = $fnet, delta w1 = $deltaw1, delta w2 = $deltaw2, delta b = $deltab, w1 = $w1, w2 = $w2, b = $b")
        }
        flag = check(array)
        iter++
    }
    return arrayOf(w1, w2, b)
}

fun check(data : IntArray):Boolean{
    for (i in data){
        if (i != 1)
            return false
    }
    return true
}