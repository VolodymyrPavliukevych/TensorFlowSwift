import Foundation
import TensorFlow

let sequenceLength = 16
let dictionaryLength = 70

func pullSequence() -> (x: Tensor<Float>, y: Tensor<Int32>) {
    let input = UUID().uuidString.prefix(sequenceLength)
    let output = String(input.sorted())
    print(input, output)
    let x = input.utf8.map { Int32($0) }
    let y = output.utf8.map { Int32($0) }
    
    return (Tensor<Float>(oneHotAtIndices: Tensor<Int32>(x), depth: 100),
            Tensor<Int32>(oneHotAtIndices: Tensor<Int32>(y), depth: 100))
}

/// MARK - LSTM Data Type, combine a few inputs / outputs
struct LSTMType: Differentiable {
    var inOrOutSignal: Tensor<Float>
    var conveyor: Tensor<Float>
    var hiddenState: Tensor<Float>
}

struct LSTMCell: Layer {
    
    typealias Input = LSTMType
    typealias Output = LSTMType

    static let inputSize = 16
    static let hiddenUnitSize = 16
    static let outputSize = 16

    var inputWGate = Dense<Float>(inputSize: inputSize, outputSize: hiddenUnitSize, activation: sigmoid)
    var inputIGate = Dense<Float>(inputSize: inputSize, outputSize: hiddenUnitSize, activation: sigmoid)

    var forgetWGate = Dense<Float>(inputSize: inputSize, outputSize: hiddenUnitSize, activation: sigmoid)
    var forgetIGate = Dense<Float>(inputSize: inputSize, outputSize: hiddenUnitSize, activation: sigmoid)

    var candidateWLayer = Dense<Float>(inputSize: inputSize, outputSize: hiddenUnitSize, activation: tanh)
    var candidateILayer = Dense<Float>(inputSize: inputSize, outputSize: hiddenUnitSize, activation: tanh)

    var outputWGate = Dense<Float>(inputSize: inputSize, outputSize: hiddenUnitSize, activation: sigmoid)
    var outputIGate = Dense<Float>(inputSize: inputSize, outputSize: hiddenUnitSize, activation: sigmoid)

    var output = Dense<Float>(inputSize: inputSize, outputSize: outputSize, activation: relu)

    @differentiable
    func applied(to input: LSTMType , in context: Context) -> LSTMType {

        let iOfT = inputWGate.applied(to: input.inOrOutSignal, in: context) + inputIGate.applied(to: input.hiddenState, in: context)
        let fOfT = forgetWGate.applied(to: input.inOrOutSignal, in: context) + forgetIGate.applied(to: input.hiddenState, in: context)
        let cOfT = candidateWLayer.applied(to: input.inOrOutSignal, in: context) + candidateILayer.applied(to: input.hiddenState, in: context)

        let conveyor = matmul(fOfT, input.conveyor) + matmul(iOfT, cOfT)
        let outputOfT = outputWGate.applied(to: input.inOrOutSignal, in: context) + outputIGate.applied(to: input.hiddenState, in: context)

        let hiddenState = matmul(outputOfT, tanh(conveyor))
        let y = output.applied(to: hiddenState, in: context)
        return LSTMType(inOrOutSignal: y, conveyor: conveyor, hiddenState: hiddenState)
    }
}


var cell = LSTMCell()
let context = Context(learningPhase: .training)
let optimizer = RMSProp<LSTMCell, Float>()
let epochCount = 100

var conveyor = Tensor<Float>(zeros: TensorShape([Int32(LSTMCell.inputSize), Int32(LSTMCell.hiddenUnitSize)]))
var hiddenState = Tensor<Float>(zeros: TensorShape([Int32(LSTMCell.inputSize), Int32(LSTMCell.hiddenUnitSize)]))


let (x, y) = pullSequence()


// The training loop.
for epoch in 0..<epochCount {
    var correctGuessCount = 0
    var totalGuessCount = 0
    var totalLoss: Float = 0
    
    let (x, y) = pullSequence()
    for charIndex in 0..<Int32(sequenceLength) {
        let input = x[charIndex]
        let output = y[charIndex]
        
        let 𝛁model = cell.gradient { cell -> Tensor<Float> in

            let cellInput = LSTMType(inOrOutSignal: input, conveyor: conveyor, hiddenState: hiddenState)
            let ŷ = cell.applied(to: cellInput, in: context)
            let correctPredictions = ŷ.inOrOutSignal.argmax(squeezingAxis: 1) .== y
            correctGuessCount += Int(Tensor<Int32>(correctPredictions).sum().scalarized())
            totalGuessCount += 1 // batchSize
            
            let loss = softmaxCrossEntropy(logits: ŷ.inOrOutSignal, labels: y)
            totalLoss += loss.scalarized()
            
//            conveyor = ŷ.conveyor
//            hiddenState = ŷ.hiddenState
            
            return loss
        }
        // Update the model's differentiable variables along the gradient vector.
        optimizer.update(&cell.allDifferentiableVariables, along: 𝛁model)
    }
        let accuracy = Float(correctGuessCount) / Float(totalGuessCount)
        print("""
            [Epoch \(epoch)] \
            Loss: \(totalLoss), \
            Accuracy: \(correctGuessCount)/\(totalGuessCount) (\(accuracy))
            """)
}


