package com.hakivin.uts

import java.io.BufferedReader
import java.io.FileReader
import kotlin.math.max

fun main() {
    //prepare data set
    val trainSize = 50
    val testSize = 5
    var totalw0 = 0.0
    var totalw1 = 0.0
    var totalw2 = 0.0
    var totalw3 = 0.0
    var totalw4 = 0.0
    var totalw5 = 0.0
    val startTime = System.nanoTime()
    var list = arrayListOf<IntArray>()
    val targetTrain = DoubleArray(trainSize)
    //ten fold cross validation
    for (time in 0..9) {
        var counter = 0
        val reader = BufferedReader(FileReader("src/dataset.txt"))
        val dataTrain = arrayListOf<IntArray>()
        val dataTest = arrayListOf<IntArray>()
        val targetTest = DoubleArray(testSize)
        for (line in reader.lines()) {
            val arr = line.split('\t')
            val score = intArrayOf(arr[0].toInt(), arr[1].toInt(), arr[2].toInt(), arr[3].toInt(), arr[4].toInt(), arr[5].toInt())
            dataTrain.add(score)
            targetTrain[counter] = arr[6].toDouble()
            if (counter >= time * 5 && counter <= time * 5 + 4) {
                dataTest.add(score)
                targetTest[counter - (time * 5)] = arr[6].toDouble()
            }
            counter++
        }
        list = dataTrain

        //input
        var w0 = Math.random() / 6
        var w1 = Math.random() / 6
        var w2 = Math.random() / 6
        var w3 = Math.random() / 6
        var w4 = Math.random() / 6
        var w5 = Math.random() / 6
        val learningRate = 0.0000001
        val tolerance = 0.000001
        val epoch = 1000000

        //train
        var flag = false
        var iter = 0
        var deltaw0: Double
        var deltaw1: Double
        var deltaw2: Double
        var deltaw3: Double
        var deltaw4: Double
        var deltaw5: Double
        while (!flag && iter <= epoch) {
            var still = true
            for (i in dataTrain.indices) {
                if (i < time * 5 || i > time * 5 + 4) {
                    val arr = dataTrain[i]
                    val target = targetTrain[i]
                    val net = w0 * arr[0] + w1 * arr[1] + w2 * arr[2] + w3 * arr[3] + w4 * arr[4] + w5 * arr[5]

                    //activation fun -> ReLU
                    val fnet = max(net, 0.0)

                    //if error is higher than tolerance, update weights
                    if (Math.abs(fnet - target) >= tolerance) {
                        //calculate delta
                        deltaw0 = learningRate * (target - fnet) * arr[0]
                        deltaw1 = learningRate * (target - fnet) * arr[1]
                        deltaw2 = learningRate * (target - fnet) * arr[2]
                        deltaw3 = learningRate * (target - fnet) * arr[3]
                        deltaw4 = learningRate * (target - fnet) * arr[4]
                        deltaw5 = learningRate * (target - fnet) * arr[5]

                        //update weight
                        w0 += deltaw0
                        w1 += deltaw1
                        w2 += deltaw2
                        w3 += deltaw3
                        w4 += deltaw4
                        w5 += deltaw5

                        still = false
                    }
                } else {
                    continue
                }
            }
            iter++
            flag = still
        }

        println("total epoch $iter")

        println("w0 =$w0, w1 = $w1, w2 = $w2, w3 = $w3, w4 = $w4, w5 = $w5")

        totalw0 += w0
        totalw1 += w1
        totalw2 += w2
        totalw3 += w3
        totalw4 += w4
        totalw5 += w5
        //Test
        println("Test result")
        for (j in dataTest.indices) {
            val arr = dataTest[j]
            val x0 = arr[0]
            val x1 = arr[1]
            val x2 = arr[2]
            val x3 = arr[3]
            val x4 = arr[4]
            val x5 = arr[5]
            val target = targetTest[j]
            val fnet = w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + w5 * x5
            println("fnet = $fnet, target = $target, error = ${(target - fnet) / target * 100} %")
        }
    }
    val finishTime = System.nanoTime()

    totalw0 /= 10
    totalw1 /= 10
    totalw2 /= 10
    totalw3 /= 10
    totalw4 /= 10
    totalw5 /= 10

    println()
    println("total time = ${(finishTime - startTime) / 1000000} ms")

    println()
    println("Final weight")
    println("w0 = $totalw0, w1 = $totalw1, w2 = $totalw2, w3 = $totalw3, w4 = $totalw4, w5 = $totalw5")

    println()
    println("Validation checking")
    for (index in list.indices) {
        val test = list[index]
        val target = targetTrain[index]

        val fnet = totalw0 * test[0] + totalw1 * test[1] + totalw2 * test[2] + totalw3 * test[3] + totalw4 * test[4] + totalw5 * test[5]
        println("fnet = $fnet, target = $target, error = ${(target - fnet) / target * 100} %")
    }
}
