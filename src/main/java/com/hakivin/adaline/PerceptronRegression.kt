package com.hakivin.adaline

import java.io.BufferedReader
import java.io.FileReader
import kotlin.math.max

fun main(){
    //prepare data set
    val trainSize = 45
    val testSize = 5
    val reader = BufferedReader(FileReader("src/dataset.txt"))
    val dataTrain = arrayListOf<IntArray>()
    val targetTrain = DoubleArray(trainSize)
    val dataTest = arrayListOf<IntArray>()
    val targetTest = DoubleArray(testSize)
    var index = 0
    for (line in reader.lines()){
        val arr = line.split('\t')
        val score = intArrayOf(arr[0].toInt(), arr[1].toInt(), arr[2].toInt(), arr[3].toInt(), arr[4].toInt(), arr[5].toInt())
        if (index < trainSize) {
            dataTrain.add(score)
            targetTrain[index] = arr[6].toDouble()
        } else {
            dataTest.add(score)
            targetTest[index-trainSize] = arr[6].toDouble()
        }
        index++
    }

    //input
    var w0 = Math.random()/6
    var w1 = Math.random()/6
    var w2 = Math.random()/6
    var w3 = Math.random()/6
    var w4 = Math.random()/6
    var w5 = Math.random()/6
    val learningRate = 0.000001
    val tolerance = 0.0001
    val epoch = 200
    println("w0 = $w0, w1 = $w1, w2 = $w2, w3 = $w3, w4 = $w4, w5 = $w5")

    //train
    var flag = false
    var iter = 0
    var deltaw0 = 0.0
    var deltaw1 = 0.0
    var deltaw2 = 0.0
    var deltaw3 = 0.0
    var deltaw4 = 0.0
    var deltaw5 = 0.0
    while (!flag && iter <= epoch){
        for (i in dataTrain.indices){
            val arr = dataTrain[i]
            val target = targetTrain[i]
            val net = w0*arr[0] + w1*arr[1] + w2*arr[2] + w3*arr[3] + w4*arr[4] + w5*arr[5]

            //activation fun -> ReLU
            val fnet = max(net,0.0)

            //calculate delta
            deltaw0 = learningRate * (target-fnet) * arr[0]
            deltaw1 = learningRate * (target-fnet) * arr[1]
            deltaw2 = learningRate * (target-fnet) * arr[2]
            deltaw3 = learningRate * (target-fnet) * arr[3]
            deltaw4 = learningRate * (target-fnet) * arr[4]
            deltaw5 = learningRate * (target-fnet) * arr[5]

            //update weight
            w0 += deltaw0
            w1 += deltaw1
            w2 += deltaw2
            w3 += deltaw3
            w4 += deltaw4
            w5 += deltaw5
            if (iter % 10 == 0) {
                println("epoch $iter")
                println("fnet = $fnet, target = $target, w0 = $w0, w1 = $w1, w2 = $w2, w3 = $w3, w4 = $w4, w5 = $w5")
            }
            iter++
        }
        if (max(deltaw0, max(deltaw1, max(deltaw2, max(deltaw3, max(deltaw4, deltaw5))))) < tolerance) {
            flag = true
        }
    }

    //Test
    println("Test result")
    for (j in dataTest.indices){
        val arr = dataTest[j]
        val x0 = arr[0]
        val x1 = arr[1]
        val x2 = arr[2]
        val x3 = arr[3]
        val x4 = arr[4]
        val x5 = arr[5]
        val target = targetTest[j]
        val fnet = w0*x0 + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5
        println("fnet = $fnet, target = $target, error = ${(target - fnet)/target*100} %")
    }
}
