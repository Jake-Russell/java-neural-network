package uk.co.claritysoftware.neuralnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuralNetwork {

  private static final Random RANDOM = new Random();
  private static final Double LEARNING_RATE = 0.1;

  private Integer numberOfInputs;
  private Integer numberOfHiddenLayers;

  private Double[][] inputsToHiddenLayerWeighting;

  private Double[] hiddenLayerBiases;
  private Double[] hiddenLayersToOutputWeighting;

  private Double[] hiddenLayerOutputs;
  private Double[] hiddenLayerDeltas;

  private Double outputLayerBias;

  private Double standardizedOutput;
  private Double deStandardizedOutput;
  private Double outputDelta;

  private Integer[] inputMins;
  private Integer[] inputMaxs;
  private Integer expectedMin;
  private Integer expectedMax;

  public static void main(String[] args) {

    NeuralNetwork network = new NeuralNetwork(2, 2);

    List<TrainingData> trainingDataList = new ArrayList<>();
    for (int n = 0; n < 300; n++) {
      Integer input1 = RANDOM.nextInt(100) + 1;
      Integer input2 = RANDOM.nextInt(100) + 1;
      Integer expectedOutput = input1 + input2;
      trainingDataList.add(new TrainingData(input1, input2, expectedOutput));
    }
    List<TrainingData> validationDataList = new ArrayList<>();
    for (int n = 0; n < 100; n++) {
      Integer input1 = RANDOM.nextInt(100) + 1;
      Integer input2 = RANDOM.nextInt(100) + 1;
      Integer expectedOutput = input1 + input2;
      validationDataList.add(new TrainingData(input1, input2, expectedOutput));
    }
    List<TrainingData> allData = new ArrayList<>(trainingDataList);
    allData.addAll(validationDataList);
    network.calculateMinsAndMaxes(allData);

    network.train(trainingDataList, validationDataList);

    network.predict(new Integer[] {13, 4});
    System.out.println("Prediction: " + network.deStandardizedOutput);

  }

  public NeuralNetwork(Integer numberOfInputs, Integer numberOfHiddenLayers) {
    this.numberOfInputs = numberOfInputs;
    this.numberOfHiddenLayers = numberOfHiddenLayers;

    this.inputsToHiddenLayerWeighting = inputsToHiddenLayerWeighting();

    this.hiddenLayerBiases = hiddenLayerBiases();
    this.hiddenLayersToOutputWeighting = hiddenLayersToOutputWeighting();

    this.outputLayerBias = outputLayerBias();

    this.hiddenLayerOutputs = new Double[numberOfHiddenLayers];
    this.hiddenLayerDeltas = new Double[numberOfHiddenLayers];
  }

  public void predict(Integer[] inputs) {
    calculateOutput(inputs);
  }

  public void train(List<TrainingData> trainingDataList, List<TrainingData> verificationDataList) {

    Integer iterationCount = 0;

    boolean carryOnTraining = true;
    Double previousRootMeanSquaredError = Double.MAX_VALUE;

    while(carryOnTraining) {


      for (int i = 0; i < 500; i++) {
        iterationCount++;

        // use all training data values
        for (TrainingData trainingData : trainingDataList) {

          // forward pass
          calculateOutput(new Integer[] { trainingData.getInput1(), trainingData.getInput2() });
          calculateOutputDelta(trainingData.getExpectedOutput());
          calculateHiddenLayerDeltas();

          // reverse pass
          this.inputsToHiddenLayerWeighting = recalculateInputsToHiddenLayerWeighting(new Integer[] { trainingData.getInput1(), trainingData.getInput2() });
          this.hiddenLayerBiases = recalculateHiddenLayerBiases();
          this.hiddenLayersToOutputWeighting = recalculateHiddenLayersToOutputWeighting();
          this.outputLayerBias = recalculateOutputLayerBias();
        }
      }

      // run all verification values through
      Double squaredError = 0.0;
      for (TrainingData verificationData : verificationDataList) {

        // forward pass
        calculateOutput(new Integer[] { verificationData.getInput1(), verificationData.getInput2() });
        calculateOutputDelta(verificationData.getExpectedOutput());
        calculateHiddenLayerDeltas();

        // reverse pass
        this.inputsToHiddenLayerWeighting = recalculateInputsToHiddenLayerWeighting(new Integer[] { verificationData.getInput1(), verificationData.getInput2() });
        this.hiddenLayerBiases = recalculateHiddenLayerBiases();
        this.hiddenLayersToOutputWeighting = recalculateHiddenLayersToOutputWeighting();
        this.outputLayerBias = recalculateOutputLayerBias();

        deStandardizedOutput = destandardizedValue(standardizedOutput, expectedMin, expectedMax);

        squaredError = squaredError + Math.pow(verificationData.getExpectedOutput() - deStandardizedOutput, 2);
      }

      //System.out.println("SE " + squaredError);
      Double rootMeanSquaredError = Math.sqrt(squaredError / verificationDataList.size());
      //System.out.println("RMSE " + rootMeanSquaredError);
      if (rootMeanSquaredError > previousRootMeanSquaredError) {
        carryOnTraining = false;
      } else {
        previousRootMeanSquaredError = rootMeanSquaredError;
      }

    }

    System.out.println("Training done with " + iterationCount + " iterations");





  }

  private void calculateMinsAndMaxes(List<TrainingData> allData) {
    inputMins = new Integer[numberOfInputs];
    inputMins[0] = allData.stream()
      .map(trainingData -> trainingData.getInput1())
      .min(Integer::compareTo)
      .orElse(0);
    inputMins[1] = allData.stream()
      .map(trainingData -> trainingData.getInput2())
      .min(Integer::compareTo)
      .orElse(0);
    inputMaxs = new Integer[numberOfInputs];
    inputMaxs[0] = allData.stream()
      .map(trainingData -> trainingData.getInput1())
      .max(Integer::compareTo)
      .orElse(0);
    inputMaxs[1] = allData.stream()
      .map(trainingData -> trainingData.getInput2())
      .max(Integer::compareTo)
      .orElse(0);
    expectedMin = allData.stream()
      .map(trainingData -> trainingData.getExpectedOutput())
      .min(Integer::compareTo)
      .orElse(0);
    expectedMax = allData.stream()
      .map(trainingData -> trainingData.getExpectedOutput())
      .max(Integer::compareTo)
      .orElse(0);
  }

  private void calculateHiddenLayerDeltas() {
    for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenLayers; hiddenLayerNum++) {
      Double firstDifferential = hiddenLayerOutputs[hiddenLayerNum] * (1 - hiddenLayerOutputs[hiddenLayerNum]);
      Double delta = hiddenLayersToOutputWeighting[hiddenLayerNum] * outputDelta * firstDifferential;
      hiddenLayerDeltas[hiddenLayerNum] = delta;
    }
  }

  private void calculateOutputDelta(int expectedValue) {
    double standardisedExpectedValue = standardizedValue(expectedValue, expectedMin, expectedMax);
    double firstDifferential = standardizedOutput * (1 - standardizedOutput);
    outputDelta = (standardisedExpectedValue - standardizedOutput) * firstDifferential;
  }

  private void calculateOutput(Integer[] inputs) {

    Double[] standardisedInputs = new Double[numberOfInputs];
    for (int inputNum = 0; inputNum < numberOfInputs; inputNum++) {
      Double standardisedValue = standardizedValue(inputs[inputNum], inputMins[inputNum], inputMaxs[inputNum]);
      standardisedInputs[inputNum] = standardisedValue;
    }

    calculateHiddenLayerValues(standardisedInputs);

    Double weightedSum = 0.0;
    for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenLayers; hiddenLayerNum++) {
      weightedSum = weightedSum + (hiddenLayerOutputs[hiddenLayerNum] * hiddenLayersToOutputWeighting[hiddenLayerNum]);
    }
    weightedSum = weightedSum + outputLayerBias;

    standardizedOutput = sigmoidFunction(weightedSum);
    deStandardizedOutput = destandardizedValue(standardizedOutput, expectedMin, expectedMax);
  }

  private void calculateHiddenLayerValues(Double[] inputs) {
    for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenLayers; hiddenLayerNum++) {

      Double weightedSum = 0.0;
      for (int inputNum = 0; inputNum < numberOfInputs; inputNum++) {
        weightedSum = weightedSum + (inputs[inputNum] * inputsToHiddenLayerWeighting[inputNum][hiddenLayerNum]);
      }
      weightedSum = weightedSum + hiddenLayerBiases[hiddenLayerNum];

      hiddenLayerOutputs[hiddenLayerNum] = sigmoidFunction(weightedSum);
     }
  }

  private Double sigmoidFunction(Double weightedSum) {
    return 1 / (1 + Math.exp(weightedSum * -1));
  }

  private Double unSigmoidFunction(Double standardizedValue) {
    return Math.log(standardizedValue / (1 - standardizedValue));
  }

  private Double[] hiddenLayerBiases() {
    Double[] hiddenLayerBiases = new Double[numberOfHiddenLayers];

    for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenLayers; hiddenLayerNum++) {
      hiddenLayerBiases[hiddenLayerNum] = randomNumber(numberOfInputs);
    }

    return hiddenLayerBiases;
  }

  private Double[] recalculateHiddenLayerBiases() {
    Double[] hiddenLayerBiases = new Double[numberOfHiddenLayers];

    for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenLayers; hiddenLayerNum++) {
      Double newBias = this.hiddenLayerBiases[hiddenLayerNum] + (LEARNING_RATE * this.hiddenLayerDeltas[hiddenLayerNum] * 1);
      hiddenLayerBiases[hiddenLayerNum] = newBias;
    }

    return hiddenLayerBiases;

  }

  private Double outputLayerBias() {
    return randomNumber(numberOfHiddenLayers);
  }

  private Double recalculateOutputLayerBias() {
    return outputLayerBias + (LEARNING_RATE * outputDelta * 1);
  }

  private Double[][] inputsToHiddenLayerWeighting() {
    Double[][] inputsToHiddenLayerWeighting = new Double[numberOfInputs][numberOfHiddenLayers];

    for (int inputNum = 0; inputNum < numberOfInputs; inputNum++) {
      inputsToHiddenLayerWeighting[inputNum] = randomWeightings();
    }

    return inputsToHiddenLayerWeighting;
  }

  private Double[][] recalculateInputsToHiddenLayerWeighting(Integer[] inputValues) {
    Double[][] inputsToHiddenLayerWeighting = new Double[numberOfInputs][numberOfHiddenLayers];

    for (int inputNum = 0; inputNum < numberOfInputs; inputNum++) {

      for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenLayers; hiddenLayerNum++) {
        Double newWeight = this.inputsToHiddenLayerWeighting[inputNum][hiddenLayerNum] + (LEARNING_RATE * hiddenLayerDeltas[hiddenLayerNum] * inputValues[inputNum]);
        inputsToHiddenLayerWeighting[inputNum][hiddenLayerNum] = newWeight;
      }
    }

    return inputsToHiddenLayerWeighting;
  }

  private Double[] hiddenLayersToOutputWeighting() {
    Double[] hiddenLayersToOutputWeighting = new Double[numberOfHiddenLayers];

    for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenLayers; hiddenLayerNum++) {
      hiddenLayersToOutputWeighting[hiddenLayerNum] = randomNumber(numberOfHiddenLayers);
    }

    return hiddenLayersToOutputWeighting;
  }

  private Double[] recalculateHiddenLayersToOutputWeighting() {
    Double[] hiddenLayersToOutputWeighting = new Double[numberOfHiddenLayers];

    for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenLayers; hiddenLayerNum++) {
      Double newWeight = this.hiddenLayersToOutputWeighting[hiddenLayerNum] + (LEARNING_RATE * outputDelta * hiddenLayerOutputs[hiddenLayerNum]);
      hiddenLayersToOutputWeighting[hiddenLayerNum] = newWeight;
    }

    return hiddenLayersToOutputWeighting;
  }

  private Double[] randomWeightings() {
    Double[] randomWeightings = new Double[numberOfHiddenLayers];

    for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenLayers; hiddenLayerNum++) {
      randomWeightings[hiddenLayerNum] = randomNumber(numberOfInputs);
    }

    return  randomWeightings;
  }

  private Double randomNumber(Integer extent) {
    int min = -2 / extent;
    int max = 2 / extent;
    return min + (max - min) * RANDOM.nextDouble();
  }

  private double standardizedValue(int value, int rangeMin, int rangeMax) {
    return 0.8 * (
      (double)(value - rangeMin) / (double)(rangeMax - rangeMin)
    ) + 0.1;
  }

  private double destandardizedValue(double value, int rangeMin, int rangeMax) {
    return (((value - 0.1) / 0.8) * (rangeMax - rangeMin)) + rangeMin;
  }
}
