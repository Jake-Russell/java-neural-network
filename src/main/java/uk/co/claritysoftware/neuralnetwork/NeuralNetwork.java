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
  private Double learningRate;
  private Double momentumTerm;
  private ActivationFunctions activationFunctions;

  private Double[][] inputsToHiddenLayerWeighting;

  private Double[] hiddenLayerBiases;
  private Double[] hiddenLayerToOutputWeighting;

  private Double[] hiddenLayerOutputs;
  private Double[] hiddenLayerDeltas;

  private Double outputLayerBias;

  private Double output;
  private Double outputDelta;


  public NeuralNetwork(Integer numberOfInputs, Integer numberOfHiddenLayers, Double learningRate, ActivationFunctions activationFunctions) {
    this.numberOfInputs = numberOfInputs;
    this.numberOfHiddenLayers = numberOfHiddenLayers;
    this.learningRate = learningRate;
    this.activationFunctions = activationFunctions;

    this.inputsToHiddenLayerWeighting = inputsToHiddenLayerWeighting();

    this.hiddenLayerBiases = hiddenLayerBiases();
    this.hiddenLayerToOutputWeighting = hiddenLayerToOutputWeighting();

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
    List<String[]> networkPerformanceData = new ArrayList<>();

    boolean carryOnTraining = true;
    Double previousRootMeanSquaredError = Double.MAX_VALUE;

    Double squaredErrorTraining = 0.0;
    while(carryOnTraining) {
    //while(epochCount<50000) {

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
      Double rootMeanSquaredErrorTraining = Math.sqrt(squaredErrorTraining / trainingDataList.size());
      meanSquaredErrorDataTraining.add(rootMeanSquaredErrorTraining);


      // run all validation values through
      Double squaredError = 0.0;
      for (CatchmentArea validationData : validationDataList) {

        // forward pass
        calculateOutput(new Double[] { validationData.getArea(), validationData.getBaseFlowIndex(), validationData.getFloodAttenuation(), validationData.getFloodPlainExtent(),
                validationData.getLongestDrainagePath(), validationData.getProportionWetDays(), validationData.getMedianAnnualMax1DayRainfall(), validationData.getStandardAnnualAverageRainfall() });

        squaredError = squaredError + Math.pow(validationData.getIndexFlood() - output, 2);
      }

      //System.out.println("SE " + squaredError);
      Double rootMeanSquaredError = Math.sqrt(squaredError / validationDataList.size());
      Double meanSquaredError = squaredError / validationDataList.size();
      // System.out.println("MSE of Validation Data is " + meanSquaredError);
      // System.out.println("RMSE of Validation Data is " + rootMeanSquaredError);
      meanSquaredErrorData.add(rootMeanSquaredError);
      epochNumberData.add(epochCount);


      if (rootMeanSquaredError > previousRootMeanSquaredError) {
        carryOnTraining = false;

/*
        previousRootMeanSquaredError = rootMeanSquaredError;
        // Test to see if stuck in local minimum

        for(int j = 0; j < 50; j++){
          for (int i = 0; i < 500; i++) {
            epochCount++;
            // use all training data values
            for (CatchmentArea trainingData : trainingDataList) {
              // forward pass
              calculateOutput(new Double[] { trainingData.getArea(), trainingData.getBaseFlowIndex(), trainingData.getFloodAttenuation(), trainingData.getFloodPlainExtent(),
                      trainingData.getLongestDrainagePath(), trainingData.getProportionWetDays(), trainingData.getMedianAnnualMax1DayRainfall(), trainingData.getStandardAnnualAverageRainfall() });


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
          // run all validation values through
          squaredError = 0.0;
          for (CatchmentArea validationData : validationDataList) {

            // forward pass
            calculateOutput(new Double[] { validationData.getArea(), validationData.getBaseFlowIndex(), validationData.getFloodAttenuation(), validationData.getFloodPlainExtent(),
                    validationData.getLongestDrainagePath(), validationData.getProportionWetDays(), validationData.getMedianAnnualMax1DayRainfall(), validationData.getStandardAnnualAverageRainfall() });

            squaredError = squaredError + Math.pow(validationData.getIndexFlood() - output, 2);
          }

          //System.out.println("SE " + squaredError);
          rootMeanSquaredError = Math.sqrt(squaredError / validationDataList.size());
          meanSquaredError = squaredError / validationDataList.size();
          // System.out.println("MSE of Validation Data is " + meanSquaredError);
          // System.out.println("RMSE of Validation Data is " + rootMeanSquaredError);
          meanSquaredErrorData.add(rootMeanSquaredError);
          epochNumberData.add(epochCount);
        }


        if(rootMeanSquaredError > previousRootMeanSquaredError){
          carryOnTraining = false;
        }
*/

      }

      else {
        previousRootMeanSquaredError = rootMeanSquaredError;
      }
    }

    System.out.println("\nFinished training using:" +
            "\n  - Hidden Layers = " + this.numberOfHiddenLayers +
            "\n  - Learning Rate = " + this.learningRate +
            "\n  - Number of Epochs = " + epochCount +
            "\n  - Activation Function = " + this.activationFunctions.toString());

    try{
      File csvFile = new File("networkPerformance.csv");
      File csvFile2 = new File("networkPerformanceTraining.csv");
      PrintWriter out = new PrintWriter(csvFile);
      PrintWriter out2 = new PrintWriter(csvFile2);
      for(int i=0; i<meanSquaredErrorData.size(); i++){
        out.println(epochNumberData.get(i) + ", " + meanSquaredErrorData.get(i));
      }
      out.close();

      for(int i=0; i<meanSquaredErrorDataTraining.size(); i++){
        out2.println(epochNumberData.get(i) + ", " + meanSquaredErrorDataTraining.get(i));
      }
      out2.close();


    } catch (FileNotFoundException e){
      System.out.println("File not found");
    }
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
      Double newBias = this.hiddenLayerBiases[hiddenLayerNum] + (learningRate * this.hiddenLayerDeltas[hiddenLayerNum] * 1);
      hiddenLayerBiases[hiddenLayerNum] = newBias;
    }

    return hiddenLayerBiases;

  }

  private Double outputLayerBias() {
    return randomNumber(numberOfHiddenLayers);
  }

  private Double recalculateOutputLayerBias() {
    return outputLayerBias + (learningRate * outputDelta * 1);
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
        Double newWeight = this.inputsToHiddenLayerWeighting[inputNum][hiddenLayerNum] + (learningRate * hiddenLayerDeltas[hiddenLayerNum] * inputValues[inputNum]);
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
    Double[] hiddenLayersToOutputWeighting = new Double[numberOfHiddenLayers];

    for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenLayers; hiddenLayerNum++) {
      Double newWeight = this.hiddenLayerToOutputWeighting[hiddenLayerNum] + (learningRate * outputDelta * hiddenLayerOutputs[hiddenLayerNum]);
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

  private Double randomNumber(double extent) {
    double min = -2 / extent;
    double max = 2 / extent;
    return min + (max - min) * RANDOM.nextDouble();
  }
}
