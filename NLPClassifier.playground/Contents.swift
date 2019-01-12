import CreateML
import Cocoa
let data = try MLDataTable(contentsOf: URL(fileURLWithPath:"/Users/Carlos/Desktop/twitter-sanders-apple3.csv" ))

let (training_data,test_data) = data.randomSplit(by: 0.9)
let classifier = try MLTextClassifier(trainingData: training_data, textColumn: "text", labelColumn: "class")

let metrics = classifier.evaluation(on: test_data)

metrics.classificationError

try classifier.prediction(from: "this tutorial was awesome")

try classifier.write(toFile: "Enter your destination file here")

