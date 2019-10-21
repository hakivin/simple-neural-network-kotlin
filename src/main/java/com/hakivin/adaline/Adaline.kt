package com.hakivin.adaline

import java.io.BufferedReader
import java.io.FileReader
import kotlin.math.max

fun main(){
    //prepare data set
    val reader = BufferedReader(FileReader("src/dataset.txt"))
    val dataTrain = arrayListOf<IntArray>()
    val targetTrain = DoubleArray(25)
    val dataTest = arrayListOf<IntArray>()
    val targetTest = DoubleArray(5)
    var index = 0
    for (line in reader.lines()){
        val arr = line.split('\t')
        val score = intArrayOf(arr[0].toInt(), arr[1].toInt(), arr[2].toInt())
        if (index < 25) {
            dataTrain.add(score)
            targetTrain[index] = arr[3].toDouble()
        } else {
            dataTest.add(score)
            targetTest[index-25] = arr[3].toDouble()
        }
        index++
    }

    //input
    var w0 = Math.random()/2
    var w1 = Math.random()/2
    var w2 = Math.random()/2
    var b = 0.0
    val learningRate = 0.0001
    val tolerance = 0.001
    val epoch = 200

    //train
    var flag = false
    var iter = 0
    var deltaw0 = 0.0
    var deltaw1 = 0.0
    var deltaw2 = 0.0
    var deltab : Double
    while (!flag && iter <= epoch){
        for (i in dataTrain.indices){
            val arr = dataTrain[i]
            val target = targetTrain[i]
            val net = w0*arr[0] + w1*arr[1] + w2*arr[2] + b

            //activation fun -> ReLU
            val fnet = when {
                net > 0 -> net
                net < 0 -> 0.0
                else -> net
            }

            //calculate delta
            deltaw0 = learningRate * (target-fnet) * arr[0]
            deltaw1 = learningRate * (target-fnet) * arr[1]
            deltaw2 = learningRate * (target-fnet) * arr[2]
            deltab = learningRate * target

            //update weight
            w0 += deltaw0
            w1 += deltaw1
            w2 += deltaw2
            b += deltab
            if (iter % 10 == 0) {
                println("epoch $iter")
                println("fnet = $fnet, target = $target, w0 = $w0, w1 = $w1, w2 = $w2, b = $b, deltaw0 = $deltaw0, deltaw1 = $deltaw1, deltaw2 = $deltaw2, deltab = $deltab")
            }
            iter++
        }
        if (max(deltaw0, max(deltaw1, deltaw2)) < tolerance)
            flag = true
    }

    //Test
    println("Test result")
    for (j in dataTest.indices){
        val arr = dataTest[j]
        val x0 = arr[0]
        val x1 = arr[1]
        val x2 = arr[2]
        val target = targetTest[j]
        val fnet = w0*x0 + w1*x1 + w2*x2 + b
        println("fnet = $fnet, target = $target, error = ${(target - fnet)/target*100}")
    }
}
