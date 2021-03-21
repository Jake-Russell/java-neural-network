package uk.co.claritysoftware.neuralnetwork;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuralNetwork {

  private static final Random RANDOM = new Random();

  private Integer numberOfInputs;
  private Integer numberOfHiddenLayers;
  private double learningRate;
  private double momentumTerm = 0.9;
  private ActivationFunctions activationFunctions;

  private Double[][] inputsToHiddenLayerWeighting;
  private Double[][] previousInputsToHiddenLayerWeighting;

  private Double[] hiddenLayerBiases;
  private Double[] previousHiddenLayerBiases;
  private Double[] hiddenLayerToOutputWeighting;
  private Double[] previousHiddenLayerToOutputWeighting;

  private Double[] hiddenLayerOutputs;
  private Double[] hiddenLayerDeltas;

  private Double outputLayerBias;
  private Double previousOutputLayerBias;

  private Double output;
  private Double outputDelta;


  public NeuralNetwork(Integer numberOfInputs, Integer numberOfHiddenLayers, Double learningRate, ActivationFunctions activationFunctions) {
    this.numberOfInputs = numberOfInputs;
    this.numberOfHiddenLayers = numberOfHiddenLayers;
    this.learningRate = learningRate;
    this.activationFunctions = activationFunctions;

    this.inputsToHiddenLayerWeighting = inputsToHiddenLayerWeighting();
    this.previousInputsToHiddenLayerWeighting = new Double[numberOfInputs][numberOfHiddenLayers];

    this.hiddenLayerBiases = hiddenLayerBiases();
    this.previousHiddenLayerBiases = new Double[numberOfHiddenLayers];
    this.hiddenLayerToOutputWeighting = hiddenLayerToOutputWeighting();
    this.previousHiddenLayerToOutputWeighting = new Double[numberOfHiddenLayers];

    this.outputLayerBias = outputLayerBias();

    this.hiddenLayerOutputs = new Double[numberOfHiddenLayers];
    this.hiddenLayerDeltas = new Double[numberOfHiddenLayers];
  }

  public double predict(CatchmentArea testData) {
    calculateOutput(new Double[] { testData.getArea(), testData.getBaseFlowIndex(), testData.getFloodAttenuation(), testData.getFloodPlainExtent(),
            testData.getLongestDrainagePath(), testData.getProportionWetDays(), testData.getMedianAnnualMax1DayRainfall(), testData.getStandardAnnualAverageRainfall() });
    return output;
  }

  public void train(List<CatchmentArea> trainingDataList, List<CatchmentArea> validationDataList) {

    Integer epochCount = 0;
    List<Double> meanSquaredErrorData = new ArrayList<>();
    List<Double> meanSquaredErrorDataTraining = new ArrayList<>();
    List<Integer> epochNumberData = new ArrayList<>();
    List<Double> learningRateData = new ArrayList<>();

    boolean carryOnTraining = true;
    double startingLearningRate = this.learningRate;
    double endingLearningRate = 0.01;
    double previousRootMeanSquaredError = Double.MAX_VALUE;

    double squaredErrorTraining = 0.0;
    //while(carryOnTraining) {
    while(epochCount<10000) {

      for (int i = 0; i < 500; i++) {
        epochCount++;

        // use all training data values
        squaredErrorTraining = 0.0;
        for (CatchmentArea trainingData : trainingDataList) {

          // forward pass
          calculateOutput(new Double[] { trainingData.getArea(), trainingData.getBaseFlowIndex(), trainingData.getFloodAttenuation(), trainingData.getFloodPlainExtent(),
                  trainingData.getLongestDrainagePath(), trainingData.getProportionWetDays(), trainingData.getMedianAnnualMax1DayRainfall(), trainingData.getStandardAnnualAverageRainfall() });

          squaredErrorTraining = squaredErrorTraining + Math.pow(trainingData.getIndexFlood() - output, 2);


          // reverse pass
          calculateOutputDelta(trainingData.getIndexFlood());
          calculateHiddenLayerDeltas();

          this.inputsToHiddenLayerWeighting = recalculateInputsToHiddenLayerWeighting(new Double[] { trainingData.getArea(), trainingData.getBaseFlowIndex(), trainingData.getFloodAttenuation(),
                  trainingData.getFloodPlainExtent(), trainingData.getLongestDrainagePath(), trainingData.getProportionWetDays(), trainingData.getMedianAnnualMax1DayRainfall(),
                  trainingData.getStandardAnnualAverageRainfall() });
          this.hiddenLayerBiases = recalculateHiddenLayerBiases();
          this.hiddenLayerToOutputWeighting = recalculateHiddenLayersToOutputWeighting();
          this.outputLayerBias = recalculateOutputLayerBias();
        }
      }
      double rootMeanSquaredErrorTraining = Math.sqrt(squaredErrorTraining / trainingDataList.size());
      meanSquaredErrorDataTraining.add(rootMeanSquaredErrorTraining);


      // run all validation values through
      double squaredError = 0.0;
      for (CatchmentArea validationData : validationDataList) {

        // forward pass
        calculateOutput(new Double[] { validationData.getArea(), validationData.getBaseFlowIndex(), validationData.getFloodAttenuation(), validationData.getFloodPlainExtent(),
                validationData.getLongestDrainagePath(), validationData.getProportionWetDays(), validationData.getMedianAnnualMax1DayRainfall(), validationData.getStandardAnnualAverageRainfall() });

        squaredError = squaredError + Math.pow(validationData.getIndexFlood() - output, 2);
      }

      double rootMeanSquaredError = Math.sqrt(squaredError / validationDataList.size());
      meanSquaredErrorData.add(rootMeanSquaredError);
      epochNumberData.add(epochCount);

      //this.learningRate = endingLearningRate + (startingLearningRate - endingLearningRate) * (1 - (1 / (1 + Math.exp(10 - ((20 * epochCount) / 10000)))));
      learningRateData.add(this.learningRate);


      if (rootMeanSquaredError > previousRootMeanSquaredError) {
        double rootMeanSquaredErrorDifference = rootMeanSquaredError - previousRootMeanSquaredError;
        double rootMeanSquaredErrorPercentageIncrease = (rootMeanSquaredErrorDifference/previousRootMeanSquaredError) * 100;

/*
  BOLD DRIVER ALGORITHM

        if(rootMeanSquaredErrorPercentageIncrease > 2){
          //System.out.println("RMSE increased by more than 2%.");
          if(this.learningRate * 0.7 > 0.01){
            this.undoWeightChange();
            this.learningRate = this.learningRate * 0.7;
            System.out.println("Now DECREASING the learning rate to " + this.learningRate);
          } else {
            System.out.println("Cannot DECREASE the learning rate anymore, since it is already " + this.learningRate);
            carryOnTraining = false;
          }
        } else{
          System.out.println("NOT changing learning rate, since percentage increase was too small - " + rootMeanSquaredErrorPercentageIncrease);
          carryOnTraining = false;
        }


      } else {
        previousRootMeanSquaredError = rootMeanSquaredError;
        if(this.learningRate * 1.05 < 0.5){
          this.learningRate = this.learningRate * 1.05;
          System.out.println("Now INCREASING the learning rate to " + this.learningRate);
        }
        else{
          System.out.println("Cannot INCREASE the learning rate anymore, since it is already " + this.learningRate);
        }
 */
      }
    }

    System.out.println("\nFinished training using:" +
            "\n  - Hidden Layers = " + this.numberOfHiddenLayers +
            "\n  - Learning Rate = " + this.learningRate +
            "\n  - Number of Epochs = " + epochCount +
            "\n  - Activation Function = " + this.activationFunctions.toString());

    try{
      File csvFile = new File("CSV/RMSE_Validation_Dataset.csv");
      File csvFileTraining = new File("CSV/RMSE_Training_Dataset.csv");
      File csvFileLearningRate = new File("CSV/Learning_Rate_Change.csv");
      PrintWriter out = new PrintWriter(csvFile);
      PrintWriter outTraining = new PrintWriter(csvFileTraining);
      PrintWriter outLearningRate = new PrintWriter(csvFileLearningRate);

      for(int i=0; i<meanSquaredErrorData.size(); i++){
        out.println(epochNumberData.get(i) + ", " + meanSquaredErrorData.get(i));
      }
      out.close();

      for(int i=0; i<meanSquaredErrorDataTraining.size(); i++){
        outTraining.println(epochNumberData.get(i) + ", " + meanSquaredErrorDataTraining.get(i));
      }
      outTraining.close();

      for(int i=0; i<learningRateData.size(); i++){
        outLearningRate.println(epochNumberData.get(i) + ", " + learningRateData.get(i));
      }
      outLearningRate.close();

    } catch (FileNotFoundException e){
      System.out.println("File not found");
    }
  }


  private void undoWeightChange(){
    this.inputsToHiddenLayerWeighting = this.previousInputsToHiddenLayerWeighting;
    this.hiddenLayerBiases = this.previousHiddenLayerBiases;
    this.hiddenLayerToOutputWeighting = this.previousHiddenLayerToOutputWeighting;
    this.outputLayerBias = this.previousOutputLayerBias;
  }

  private void calculateHiddenLayerDeltas() {
    for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenLayers; hiddenLayerNum++) {
      Double firstDerivative;
      switch (this.activationFunctions){
        case TANH:
          firstDerivative = firstDerivativeTanH(hiddenLayerOutputs[hiddenLayerNum]);
          break;
        case RELU:
          firstDerivative = firstDerivativeRelu(hiddenLayerOutputs[hiddenLayerNum]);
          break;
        default:
          firstDerivative = firstDerivativeSigmoid(hiddenLayerOutputs[hiddenLayerNum]);
      }
      Double delta = hiddenLayerToOutputWeighting[hiddenLayerNum] * outputDelta * firstDerivative;
      hiddenLayerDeltas[hiddenLayerNum] = delta;
    }
  }

  private void calculateOutputDelta(double expectedValue) {
    Double firstDerivative;
    switch (this.activationFunctions){
      case TANH:
        firstDerivative = firstDerivativeTanH(output);
        break;
      case RELU:
        firstDerivative = firstDerivativeRelu(output);
        break;
      default:
        firstDerivative = firstDerivativeSigmoid(output);
    }

    outputDelta = (expectedValue - output) * firstDerivative;
  }

  private void calculateOutput(Double[] inputs) {

    calculateHiddenLayerValues(inputs);

    Double weightedSum = 0.0;
    for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenLayers; hiddenLayerNum++) {
      weightedSum = weightedSum + (hiddenLayerOutputs[hiddenLayerNum] * hiddenLayerToOutputWeighting[hiddenLayerNum]);
    }
    weightedSum = weightedSum + outputLayerBias;

    switch (this.activationFunctions){
      case TANH:
        output = tanHFunction(weightedSum);
        break;
      case RELU:
        output = reluFunction(weightedSum);
        break;
      default:
        output = sigmoidFunction(weightedSum);
    }
  }

  private void calculateHiddenLayerValues(Double[] inputs) {
    for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenLayers; hiddenLayerNum++) {

      Double weightedSum = 0.0;
      for (int inputNum = 0; inputNum < numberOfInputs; inputNum++) {
        weightedSum = weightedSum + (inputs[inputNum] * inputsToHiddenLayerWeighting[inputNum][hiddenLayerNum]);
      }
      weightedSum = weightedSum + hiddenLayerBiases[hiddenLayerNum];

      switch (this.activationFunctions){
        case TANH:
          hiddenLayerOutputs[hiddenLayerNum] = tanHFunction(weightedSum);
          break;
        case RELU:
          hiddenLayerOutputs[hiddenLayerNum] = reluFunction(weightedSum);
          break;
        default:
          hiddenLayerOutputs[hiddenLayerNum] = sigmoidFunction(weightedSum);
      }
    }
  }

  private Double sigmoidFunction(Double weightedSum) {
    return 1 / (1 + Math.exp(weightedSum * -1));
  }

  private Double firstDerivativeSigmoid(Double value) {
    return value * (1 - value);
  }

  private Double tanHFunction(Double weightedSum) {
    return (Math.exp(weightedSum) - Math.exp(weightedSum * -1)) / (Math.exp(weightedSum) + Math.exp(weightedSum * -1));
  }

  private Double firstDerivativeTanH(Double value) {
    return 1 - Math.pow(value, 2);
  }

  private Double reluFunction(Double weightedSum) {
    return Math.max(weightedSum, 0.01*weightedSum);
    // Returns weightedSum if weightedSum >= 0
    // Returns 0.01 * weightedSum if weightedSum < 0
  }

  private Double firstDerivativeRelu(Double value) {
    if(value <= 0) {
      return 0.01;
    } else {
      return 1.00;
    }
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
      double newBias = this.hiddenLayerBiases[hiddenLayerNum] + (learningRate * this.hiddenLayerDeltas[hiddenLayerNum] * 1);
      double biasDifference = newBias - this.hiddenLayerBiases[hiddenLayerNum];
      double newBiasWithMomentum = newBias + (momentumTerm * biasDifference);
      this.previousHiddenLayerBiases[hiddenLayerNum] = this.hiddenLayerBiases[hiddenLayerNum];
      //hiddenLayerBiases[hiddenLayerNum] = newBiasWithMomentum;
      hiddenLayerBiases[hiddenLayerNum] = newBias;
    }
    return hiddenLayerBiases;
  }

  private Double outputLayerBias() {
    return randomNumber(numberOfHiddenLayers);
  }

  private Double recalculateOutputLayerBias() {
    double newBias = outputLayerBias + (learningRate * outputDelta * 1);
    double biasDifference = newBias - this.outputLayerBias;
    this.previousOutputLayerBias = this.outputLayerBias;
    double newBiasWithMomentum = newBias + (momentumTerm * biasDifference);
    //return newBiasWithMomentum;
    return newBias;

  }

  private Double[][] inputsToHiddenLayerWeighting() {
    Double[][] inputsToHiddenLayerWeighting = new Double[numberOfInputs][numberOfHiddenLayers];

    for (int inputNum = 0; inputNum < numberOfInputs; inputNum++) {
      inputsToHiddenLayerWeighting[inputNum] = randomWeightings();
    }
    return inputsToHiddenLayerWeighting;
  }

  private Double[][] recalculateInputsToHiddenLayerWeighting(Double[] inputValues) {
    Double[][] inputsToHiddenLayerWeighting = new Double[numberOfInputs][numberOfHiddenLayers];

    for (int inputNum = 0; inputNum < numberOfInputs; inputNum++) {

      for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenLayers; hiddenLayerNum++) {
        double newWeight = this.inputsToHiddenLayerWeighting[inputNum][hiddenLayerNum] + (learningRate * hiddenLayerDeltas[hiddenLayerNum] * inputValues[inputNum]);
        double weightDifference = newWeight - this.inputsToHiddenLayerWeighting[inputNum][hiddenLayerNum];
        double newWeightWithMomentum = newWeight + (momentumTerm * weightDifference);
        this.previousInputsToHiddenLayerWeighting[inputNum][hiddenLayerNum] = this.inputsToHiddenLayerWeighting[inputNum][hiddenLayerNum];
        //inputsToHiddenLayerWeighting[inputNum][hiddenLayerNum] = newWeightWithMomentum;
        inputsToHiddenLayerWeighting[inputNum][hiddenLayerNum] = newWeight;
      }
    }

    return inputsToHiddenLayerWeighting;
  }

  private Double[] hiddenLayerToOutputWeighting() {
    Double[] hiddenLayersToOutputWeighting = new Double[numberOfHiddenLayers];

    for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenLayers; hiddenLayerNum++) {
      hiddenLayersToOutputWeighting[hiddenLayerNum] = randomNumber(numberOfHiddenLayers);
    }

    return hiddenLayersToOutputWeighting;
  }

  private Double[] recalculateHiddenLayersToOutputWeighting() {
    Double[] hiddenLayerToOutputWeighting = new Double[numberOfHiddenLayers];

    for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenLayers; hiddenLayerNum++) {
      double newWeight = this.hiddenLayerToOutputWeighting[hiddenLayerNum] + (learningRate * outputDelta * hiddenLayerOutputs[hiddenLayerNum]);
      double weightDifference = newWeight - this.hiddenLayerToOutputWeighting[hiddenLayerNum];
      double newWeightWithMomentum = newWeight + (momentumTerm * weightDifference);
      previousHiddenLayerToOutputWeighting[hiddenLayerNum] = this.hiddenLayerToOutputWeighting[hiddenLayerNum];
      //hiddenLayerToOutputWeighting[hiddenLayerNum] = newWeightWithMomentum;
      hiddenLayerToOutputWeighting[hiddenLayerNum] = newWeight;
    }

    return hiddenLayerToOutputWeighting;
  }

  private Double[] randomWeightings() {
    Double[] randomWeightings = new Double[numberOfHiddenLayers];

    for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenLayers; hiddenLayerNum++) {
      randomWeightings[hiddenLayerNum] = randomNumber(numberOfInputs);
    }

    return  randomWeightings;
  }

  private Double randomNumber(double extent) {
    double min = -2 / extent;
    double max = 2 / extent;
    return min + (max - min) * RANDOM.nextDouble();
  }
}
