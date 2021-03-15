package uk.co.claritysoftware.neuralnetwork;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

public class NeuralNetwork {

  private static final Random RANDOM = new Random();

  private Integer numberOfInputs;
  private Integer numberOfHiddenLayers;
  private Double learningRate;

  private Double[][] inputsToHiddenLayerWeighting;

  private Double[] hiddenLayerBiases;
  private Double[] hiddenLayersToOutputWeighting;

  private Double[] hiddenLayerOutputs;
  private Double[] hiddenLayerDeltas;

  private Double outputLayerBias;

  private Double output;
  private Double outputDelta;


  public NeuralNetwork(Integer numberOfInputs, Integer numberOfHiddenLayers, Double learningRate) {
    this.numberOfInputs = numberOfInputs;
    this.numberOfHiddenLayers = numberOfHiddenLayers;
    this.learningRate = learningRate;

    this.inputsToHiddenLayerWeighting = inputsToHiddenLayerWeighting();

    this.hiddenLayerBiases = hiddenLayerBiases();
    this.hiddenLayersToOutputWeighting = hiddenLayersToOutputWeighting();

    this.outputLayerBias = outputLayerBias();

    this.hiddenLayerOutputs = new Double[numberOfHiddenLayers];
    this.hiddenLayerDeltas = new Double[numberOfHiddenLayers];
  }

  public double predict(CatchmentArea testData) {
    calculateOutput(new Double[] { testData.getArea(), testData.getBaseFlowIndex(), testData.getFloodAttenuation(), testData.getFloodPlainExtent(),
            testData.getLongestDrainagePath(), testData.getProportionWetDays(), testData.getMedianAnnualMax1DayRainfall(), testData.getStandardAnnualAverageRainfall() });
    return output;
  }

  public void train(List<CatchmentArea> trainingDataList, List<CatchmentArea> verificationDataList) {

    Integer epochCount = 0;
    List<Double> meanSquaredErrorData = new ArrayList<>();
    List<Integer> epochNumberData = new ArrayList<>();
    List<String[]> networkPerformanceData = new ArrayList<>();

    boolean carryOnTraining = true;
    Double previousRootMeanSquaredError = Double.MAX_VALUE;

    while(carryOnTraining) {
    //while(epochCount<=30000) {

        for (int i = 0; i < 500; i++) {
        epochCount++;

        // use all training data values
        for (CatchmentArea trainingData : trainingDataList) {

          // forward pass
          calculateOutput(new Double[] { trainingData.getArea(), trainingData.getBaseFlowIndex(), trainingData.getFloodAttenuation(), trainingData.getFloodPlainExtent(),
                  trainingData.getLongestDrainagePath(), trainingData.getProportionWetDays(), trainingData.getMedianAnnualMax1DayRainfall(), trainingData.getStandardAnnualAverageRainfall() });
          calculateOutputDelta(trainingData.getIndexFlood());
          calculateHiddenLayerDeltas();

          // reverse pass
          this.inputsToHiddenLayerWeighting = recalculateInputsToHiddenLayerWeighting(new Double[] { trainingData.getArea(), trainingData.getBaseFlowIndex(), trainingData.getFloodAttenuation(),
                  trainingData.getFloodPlainExtent(), trainingData.getLongestDrainagePath(), trainingData.getProportionWetDays(), trainingData.getMedianAnnualMax1DayRainfall(),
                  trainingData.getStandardAnnualAverageRainfall() });
          this.hiddenLayerBiases = recalculateHiddenLayerBiases();
          this.hiddenLayersToOutputWeighting = recalculateHiddenLayersToOutputWeighting();
          this.outputLayerBias = recalculateOutputLayerBias();
        }
      }

      // run all verification values through
      Double squaredError = 0.0;
      for (CatchmentArea verificationData : verificationDataList) {

        // forward pass
        calculateOutput(new Double[] { verificationData.getArea(), verificationData.getBaseFlowIndex(), verificationData.getFloodAttenuation(), verificationData.getFloodPlainExtent(),
                verificationData.getLongestDrainagePath(), verificationData.getProportionWetDays(), verificationData.getMedianAnnualMax1DayRainfall(), verificationData.getStandardAnnualAverageRainfall() });

        squaredError = squaredError + Math.pow(verificationData.getIndexFlood() - output, 2);


      }

      //System.out.println("SE " + squaredError);
      Double rootMeanSquaredError = Math.sqrt(squaredError / verificationDataList.size());
      Double meanSquaredError = squaredError / verificationDataList.size();
      System.out.println("MSE of Validation Data is " + meanSquaredError);
      System.out.println("RMSE of Validation Data is " + rootMeanSquaredError);
      meanSquaredErrorData.add(rootMeanSquaredError);
      epochNumberData.add(epochCount);




      if (rootMeanSquaredError > previousRootMeanSquaredError) {
        carryOnTraining = false;
      } else {
        previousRootMeanSquaredError = rootMeanSquaredError;
      }
    }

    System.out.println("\nFinished using " + numberOfHiddenLayers + " hidden layers and a learning rate of " + learningRate);
    System.out.println("Training done with " + epochCount + " epochs");

    try{
      File csvFile = new File("networkPerformance.csv");
      PrintWriter out = new PrintWriter(csvFile);
      for(int i=0; i<meanSquaredErrorData.size(); i++){
        out.println(epochNumberData.get(i) + ", " + meanSquaredErrorData.get(i));
      }
      out.close();

    } catch (FileNotFoundException e){
      System.out.println("File not found");
    }
  }

  private void calculateHiddenLayerDeltas() {
    for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenLayers; hiddenLayerNum++) {
      Double firstDifferential = hiddenLayerOutputs[hiddenLayerNum] * (1 - hiddenLayerOutputs[hiddenLayerNum]);
      Double delta = hiddenLayersToOutputWeighting[hiddenLayerNum] * outputDelta * firstDifferential;
      hiddenLayerDeltas[hiddenLayerNum] = delta;
    }
  }

  private void calculateOutputDelta(double expectedValue) {
    double firstDifferential = output * (1 - output);
    outputDelta = (expectedValue - output) * firstDifferential;
  }

  private void calculateOutput(Double[] inputs) {

    calculateHiddenLayerValues(inputs);

    Double weightedSum = 0.0;
    for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenLayers; hiddenLayerNum++) {
      weightedSum = weightedSum + (hiddenLayerOutputs[hiddenLayerNum] * hiddenLayersToOutputWeighting[hiddenLayerNum]);
    }
    weightedSum = weightedSum + outputLayerBias;

    output = sigmoidFunction(weightedSum);
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
      Double newWeight = this.hiddenLayersToOutputWeighting[hiddenLayerNum] + (learningRate * outputDelta * hiddenLayerOutputs[hiddenLayerNum]);
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
