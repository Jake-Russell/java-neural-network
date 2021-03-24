package uk.ac.lboro.jakerussell.neuralnetwork;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * NeuralNetwork is responsible for creating a Neural Network, given a network configuration.
 * It is also responsible for training a network, and using it to predict an Index Flood
 *
 * @author Jake Russell
 * @version 1.0
 * @since 22/03/2021
 */
public class NeuralNetwork {

    private static final Random RANDOM = new Random();

    private int numberOfInputs;
    private int numberOfHiddenNodes;
    private double learningRate;
    private double momentumTerm = 0.9;
    private ActivationFunctions activationFunction;
    private List<Improvements> improvements;
    private int numberOfEpochsToTrainFor;

    private Double[][] inputsToHiddenLayerWeighting;
    private Double[][] previousInputsToHiddenLayerWeighting;
    private Double[][][] batchProcessingInputsToHiddenLayerWeighting;

    private Double[] hiddenLayerBiases;
    private Double[] previousHiddenLayerBiases;
    private Double[][] batchProcessingHiddenLayerBiases;

    private Double[] hiddenLayerToOutputWeighting;
    private Double[] previousHiddenLayerToOutputWeighting;
    private Double[][] batchProcessingHiddenLayerToOutputWeighting;

    private Double[] hiddenLayerOutputs;
    private Double[] hiddenLayerDeltas;

    private Double outputLayerBias;
    private Double previousOutputLayerBias;
    private Double[] batchProcessingOutputLayerBias;

    private Double output;
    private Double outputDelta;

    /**
     * Constructor takes the network configuration as input, and configures a network to match this configuration
     *
     * @param numberOfInputs           the number of inputs for the network
     * @param numberOfHiddenNodes      the number of hidden nodes in the network
     * @param learningRate             the learning rate to be used during network training
     * @param activationFunction       the activation function to be used in the network
     * @param improvements             a list of improvements to be used in the network
     * @param numberOfEpochsToTrainFor the number of epochs the network will train for, or 0 if network training should be terminated automatically
     */
    public NeuralNetwork(Integer numberOfInputs, Integer numberOfHiddenNodes, Double learningRate, ActivationFunctions activationFunction, List<Improvements> improvements, int numberOfEpochsToTrainFor) {
        this.numberOfInputs = numberOfInputs;
        this.numberOfHiddenNodes = numberOfHiddenNodes;
        this.learningRate = learningRate;
        this.activationFunction = activationFunction;
        this.improvements = improvements;
        this.numberOfEpochsToTrainFor = numberOfEpochsToTrainFor;

        this.inputsToHiddenLayerWeighting = generateRandomStartingInputsToHiddenLayerWeighting();
        this.previousInputsToHiddenLayerWeighting = new Double[numberOfInputs][numberOfHiddenNodes];

        this.hiddenLayerBiases = generateRandomStartingHiddenLayerBiases();
        this.previousHiddenLayerBiases = new Double[numberOfHiddenNodes];
        this.hiddenLayerToOutputWeighting = generateRandomStartingHiddenLayerToOutputWeighting();
        this.previousHiddenLayerToOutputWeighting = new Double[numberOfHiddenNodes];

        this.outputLayerBias = generateRandomStartingOutputLayerBias();

        this.hiddenLayerOutputs = new Double[numberOfHiddenNodes];
        this.hiddenLayerDeltas = new Double[numberOfHiddenNodes];
    }


    /**
     * Predicts the index flood, given a catchment area
     *
     * @param testData the catchment area for which the index flood should be predicted
     * @return the predicted index flood for the given catchment area
     */
    public double predict(CatchmentArea testData) {
        calculateOutput(new Double[]{testData.getArea(), testData.getBaseFlowIndex(), testData.getFloodAttenuation(), testData.getFloodPlainExtent(),
                testData.getLongestDrainagePath(), testData.getProportionWetDays(), testData.getMedianAnnualMax1DayRainfall(), testData.getStandardAnnualAverageRainfall()});
        return output;
    }


    /**
     * Trains the network using the backpropagation algorithm, given 2 lists of catchment area
     *
     * @param trainingDataList   a list of catchment area to train the network on
     * @param validationDataList a list of catchment area to validate the network on to prevent over-training
     */
    public void train(List<CatchmentArea> trainingDataList, List<CatchmentArea> validationDataList) {
        int epochCount = 0;

        // Initialising Lists to store data in during network training. This is used for CSV File Writes so that graphs can
        // be plotted.
        List<Double> rootMeanSquaredErrorValidationDataset = new ArrayList<>();
        List<Double> rootMeanSquaredErrorTrainingDataset = new ArrayList<>();
        List<Integer> epochNumberData = new ArrayList<>();
        List<Double> learningRateData = new ArrayList<>();

        // Initialising Lists required for Batch Processing
        int batchSize = 171;
        this.batchProcessingInputsToHiddenLayerWeighting = new Double[numberOfInputs][numberOfHiddenNodes][batchSize];
        this.batchProcessingHiddenLayerToOutputWeighting = new Double[numberOfHiddenNodes][batchSize];
        this.batchProcessingHiddenLayerBiases = new Double[numberOfHiddenNodes][batchSize];
        this.batchProcessingOutputLayerBias = new Double[batchSize];

        // Initialising Learning Rate bounds for Annealing
        double startingLearningRate = this.learningRate;
        double endingLearningRate = 0.01;

        // Initialising the previous RMSE to be a very large number, so that training is not instantly stopped
        double previousRootMeanSquaredError = Double.MAX_VALUE;

        boolean carryOnTraining = true;

        // If the network has been configured with a number of epochs to train for of 0, then this means training should be
        // stopped automatically when the error of the validation dataset increases. Hence, if it is not 0, then we can disregard
        // the boolean flag carryOnTraining, as the network will be trained until the specified number of epochs, even if the
        // error of the validation dataset begins to increase.
        if (this.numberOfEpochsToTrainFor != 0) {
            carryOnTraining = false;
        }

        double squaredErrorTraining = 0.0;

        // Carry on training and validating the network while either:
        //     1 - The error on the validation dataset has not increased (if training is to be terminated automatically)
        //     2 - The number of epochs trained for has not reached the specified number to train for
        while (carryOnTraining || epochCount < this.numberOfEpochsToTrainFor) {

            // Train the network for 500 epochs before validating it
            for (int i = 0; i < 500; i++) {
                epochCount++;
                squaredErrorTraining = 0.0;
                int trainingDataNumber = 0;
                // Do a forwards and backwards pass for every catchment area in the training dataset (1 forwards and backwards pass
                // through all data in the training dataset is 1 epoch)
                for (CatchmentArea trainingData : trainingDataList) {

                    // Perform a forwards pass through the network and calculate the output
                    calculateOutput(new Double[]{trainingData.getArea(), trainingData.getBaseFlowIndex(), trainingData.getFloodAttenuation(), trainingData.getFloodPlainExtent(),
                            trainingData.getLongestDrainagePath(), trainingData.getProportionWetDays(), trainingData.getMedianAnnualMax1DayRainfall(), trainingData.getStandardAnnualAverageRainfall()});
                    squaredErrorTraining = squaredErrorTraining + Math.pow(trainingData.getIndexFlood() - output, 2);

                    // Perform a backwards pass through the network
                    calculateOutputDelta(trainingData.getIndexFlood());
                    calculateHiddenLayerDeltas();

                    // If Batch Processing is to be used, then calculate the weight changes and add these to the corresponding Batch Processing list.
                    // Otherwise, calculate the weight changes and perform them.
                    if (improvements.contains(Improvements.BATCH_PROCESSING)) {
                        appendWeightChanges(new Double[]{trainingData.getArea(), trainingData.getBaseFlowIndex(), trainingData.getFloodAttenuation(), trainingData.getFloodPlainExtent(),
                                trainingData.getLongestDrainagePath(), trainingData.getProportionWetDays(), trainingData.getMedianAnnualMax1DayRainfall(), trainingData.getStandardAnnualAverageRainfall()}, trainingDataNumber);
                    } else {
                        this.inputsToHiddenLayerWeighting = recalculateInputsToHiddenLayerWeighting(new Double[]{trainingData.getArea(), trainingData.getBaseFlowIndex(), trainingData.getFloodAttenuation(),
                                trainingData.getFloodPlainExtent(), trainingData.getLongestDrainagePath(), trainingData.getProportionWetDays(), trainingData.getMedianAnnualMax1DayRainfall(),
                                trainingData.getStandardAnnualAverageRainfall()}, trainingDataList.size());
                        this.hiddenLayerBiases = recalculateHiddenLayerBiases(trainingDataList.size());
                        this.hiddenLayerToOutputWeighting = recalculateHiddenLayerToOutputWeighting(trainingDataList.size());
                        this.outputLayerBias = recalculateOutputLayerBias(trainingDataList.size());
                    }

                    // If the batch size limit has been reached, then calculate the average weight changes and perform them.
                    if (trainingDataNumber == batchSize - 1 && improvements.contains(Improvements.BATCH_PROCESSING)) {
                        this.inputsToHiddenLayerWeighting = recalculateInputsToHiddenLayerWeighting(new Double[]{trainingData.getArea(), trainingData.getBaseFlowIndex(), trainingData.getFloodAttenuation(),
                                trainingData.getFloodPlainExtent(), trainingData.getLongestDrainagePath(), trainingData.getProportionWetDays(), trainingData.getMedianAnnualMax1DayRainfall(),
                                trainingData.getStandardAnnualAverageRainfall()}, batchSize);
                        this.hiddenLayerBiases = recalculateHiddenLayerBiases(batchSize);
                        this.hiddenLayerToOutputWeighting = recalculateHiddenLayerToOutputWeighting(batchSize);
                        this.outputLayerBias = recalculateOutputLayerBias(batchSize);
                    }

                    if (trainingDataNumber == batchSize - 1) {
                        trainingDataNumber = 0;
                    } else {
                        trainingDataNumber++;
                    }
                }
            }
            // Calculate the RMSE for the training data, and add this to the corresponding list
            double rootMeanSquaredErrorTraining = Math.sqrt(squaredErrorTraining / trainingDataList.size());
            rootMeanSquaredErrorTrainingDataset.add(rootMeanSquaredErrorTraining);


            double squaredError = 0.0;
            // Now do a forwards pass only for every catchment area in the validation dataset, calculating the error after
            // each forward pass.
            for (CatchmentArea validationData : validationDataList) {

                // Perform a forwards pass through the network and calculate the output
                calculateOutput(new Double[]{validationData.getArea(), validationData.getBaseFlowIndex(), validationData.getFloodAttenuation(), validationData.getFloodPlainExtent(),
                        validationData.getLongestDrainagePath(), validationData.getProportionWetDays(), validationData.getMedianAnnualMax1DayRainfall(), validationData.getStandardAnnualAverageRainfall()});
                squaredError = squaredError + Math.pow(validationData.getIndexFlood() - output, 2);
            }

            // Calculate the RMSE for the validation data, and add this to the corresponding list
            double rootMeanSquaredError = Math.sqrt(squaredError / validationDataList.size());
            rootMeanSquaredErrorValidationDataset.add(rootMeanSquaredError);
            epochNumberData.add(epochCount);

            // If Annealing is to be used, then recalculate the Learning Rate, and add this to the corresponding list
            if (improvements.contains(Improvements.ANNEALING)) {
                this.learningRate = endingLearningRate + (startingLearningRate - endingLearningRate) * (1 - (1 / (1 + Math.exp(10 - ((20 * epochCount) / numberOfEpochsToTrainFor)))));
            }
            learningRateData.add(this.learningRate);

            // If the calculated RMSE is greater than the previous RMSE, then this is an indication that training should be
            // terminated as we may be over-training the network on the test dataset.
            if (rootMeanSquaredError > previousRootMeanSquaredError) {
                double rootMeanSquaredErrorDifference = rootMeanSquaredError - previousRootMeanSquaredError;
                double rootMeanSquaredErrorPercentageIncrease = (rootMeanSquaredErrorDifference / previousRootMeanSquaredError) * 100;

                // If Bold Driver is to be used, the RMSE has increased by more than 2%, and updating the learning rate will
                // not take it lower than 0.01, then undo the weight change, and update the value of the learning rate
                if (improvements.contains(Improvements.BOLD_DRIVER) && rootMeanSquaredErrorPercentageIncrease > 2 && this.learningRate * 0.7 > 0.01) {
                    this.undoWeightAndBiasChanges();
                    this.learningRate = this.learningRate * 0.7;
                } else {
                    carryOnTraining = false;
                }
            } else {
                previousRootMeanSquaredError = rootMeanSquaredError;
                // If Bold Driver is to be used, the RMSE has not increased, and updating the learning rate will not take it
                // higher than 0.5, then accept the weight change, and update the value of the learning rate
                if (improvements.contains(Improvements.BOLD_DRIVER) && this.learningRate * 1.05 < 0.5) {
                    // If using Bold Driver and changing the learning rate will not take it higher than 0.5
                    this.learningRate = this.learningRate * 1.05;
                }
            }
        }

        System.out.print("\nFinished training using:" +
                "\n  - Hidden Layers = " + this.numberOfHiddenNodes +
                "\n  - Learning Rate = " + this.learningRate +
                "\n  - Number of Epochs = " + epochCount +
                "\n  - Activation Function = " + this.activationFunction.toString() +
                "\n  - Improvements = ");
        for (int improvementNumber = 0; improvementNumber < this.improvements.size(); improvementNumber++) {
            if (improvementNumber != this.improvements.size() - 1) {
                System.out.print(this.improvements.get(improvementNumber).toString() + ", ");
            } else {
                System.out.println(this.improvements.get(improvementNumber).toString());
            }
        }

        // Write to CSV Files for Graphs
        try {
            File csvFile = new File("CSV/RMSE_Validation_Dataset.csv");
            File csvFileTraining = new File("CSV/RMSE_Training_Dataset.csv");
            File csvFileLearningRate = new File("CSV/Learning_Rate_Change.csv");
            PrintWriter out = new PrintWriter(csvFile);
            PrintWriter outTraining = new PrintWriter(csvFileTraining);
            PrintWriter outLearningRate = new PrintWriter(csvFileLearningRate);

            for (int i = 0; i < rootMeanSquaredErrorValidationDataset.size(); i++) {
                out.println(epochNumberData.get(i) + ", " + rootMeanSquaredErrorValidationDataset.get(i));
            }
            out.close();

            for (int i = 0; i < rootMeanSquaredErrorTrainingDataset.size(); i++) {
                outTraining.println(epochNumberData.get(i) + ", " + rootMeanSquaredErrorTrainingDataset.get(i));
            }
            outTraining.close();

            for (int i = 0; i < learningRateData.size(); i++) {
                outLearningRate.println(epochNumberData.get(i) + ", " + learningRateData.get(i));
            }
            outLearningRate.close();

        } catch (FileNotFoundException e) {
            System.out.println("File not found");
        }
    }


    /**
     * Recalculates the weighting for the connections between the input nodes and hidden layer nodes
     *
     * @param inputValues an array of the input values for the current catchment area
     * @param batchSize   the batch size to be used in the case of batch processing
     * @return a 2D array of the updated weights for the connections between the input nodes and hidden layer nodes
     */
    private Double[][] recalculateInputsToHiddenLayerWeighting(Double[] inputValues, int batchSize) {
        Double[][] inputsToHiddenLayerWeighting = new Double[numberOfInputs][numberOfHiddenNodes];

        // For each connection from every input node to every hidden layer node
        for (int inputNum = 0; inputNum < numberOfInputs; inputNum++) {
            for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenNodes; hiddenLayerNum++) {
                double newWeight;

                // If Batch Processing is to be used, calculate the average of each weight to be updated and update the weight
                // Otherwise, just update the weight
                if (improvements.contains(Improvements.BATCH_PROCESSING)) {
                    double weightChangeSum = 0;
                    for (int iterationCount = 0; iterationCount < batchSize; iterationCount++) {
                        weightChangeSum = weightChangeSum + this.batchProcessingInputsToHiddenLayerWeighting[inputNum][hiddenLayerNum][iterationCount];
                    }
                    double averageWeightChange = weightChangeSum / batchSize;
                    newWeight = this.inputsToHiddenLayerWeighting[inputNum][hiddenLayerNum] + (learningRate * averageWeightChange);
                } else {
                    newWeight = this.inputsToHiddenLayerWeighting[inputNum][hiddenLayerNum] + (learningRate * (hiddenLayerDeltas[hiddenLayerNum] * inputValues[inputNum]));
                }

                // If Momentum is to be used, calculate the new weight value with momentum, and add this to the weight
                if (improvements.contains(Improvements.MOMENTUM)) {
                    double weightDifference = newWeight - this.inputsToHiddenLayerWeighting[inputNum][hiddenLayerNum];
                    newWeight = newWeight + (momentumTerm * weightDifference);
                }

                // Set the previous weight to the current weight, and then update the current weight
                this.previousInputsToHiddenLayerWeighting[inputNum][hiddenLayerNum] = this.inputsToHiddenLayerWeighting[inputNum][hiddenLayerNum];
                inputsToHiddenLayerWeighting[inputNum][hiddenLayerNum] = newWeight;
            }
        }
        return inputsToHiddenLayerWeighting;
    }


    /**
     * Recalculates the weighting for the connections between the hidden layer nodes and the output node
     *
     * @param batchSize the batch size to be used in the case of batch processing
     * @return an array of the updated weights for the connections between the input nodes and hidden layer nodes
     */
    private Double[] recalculateHiddenLayerToOutputWeighting(int batchSize) {
        Double[] hiddenLayerToOutputWeighting = new Double[numberOfHiddenNodes];

        // For each connection from the hidden layer nodes to the output node
        for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenNodes; hiddenLayerNum++) {
            double newWeight;

            // If Batch Processing is to be used, calculate the average of each weight to be updated and update the weight
            // Otherwise, just update the weight
            if (improvements.contains(Improvements.BATCH_PROCESSING)) {
                double weightChangeSum = 0;
                for (int iterationCount = 0; iterationCount < batchSize; iterationCount++) {
                    weightChangeSum = weightChangeSum + this.batchProcessingHiddenLayerToOutputWeighting[hiddenLayerNum][iterationCount];
                }
                double averageWeightChange = weightChangeSum / batchSize;
                newWeight = this.hiddenLayerToOutputWeighting[hiddenLayerNum] + (learningRate * averageWeightChange);

            } else {
                newWeight = this.hiddenLayerToOutputWeighting[hiddenLayerNum] + (learningRate * outputDelta * hiddenLayerOutputs[hiddenLayerNum]);
            }

            // If Momentum is to be used, calculate the new weight value with momentum, and add this to the weight
            if (improvements.contains(Improvements.MOMENTUM)) {
                double weightDifference = newWeight - this.hiddenLayerToOutputWeighting[hiddenLayerNum];
                newWeight = newWeight + (momentumTerm * weightDifference);
            }

            // Set the previous weight to the current weight, and then update the current weight
            previousHiddenLayerToOutputWeighting[hiddenLayerNum] = this.hiddenLayerToOutputWeighting[hiddenLayerNum];
            hiddenLayerToOutputWeighting[hiddenLayerNum] = newWeight;
        }
        return hiddenLayerToOutputWeighting;
    }


    /**
     * Recalculates the biases for the hidden layer nodes
     *
     * @param batchSize the batch size to be used in the case of batch processing
     * @return an array of the updated biases for the hidden layer nodes
     */
    private Double[] recalculateHiddenLayerBiases(int batchSize) {
        Double[] hiddenLayerBiases = new Double[numberOfHiddenNodes];

        // For each node in the hidden layer
        for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenNodes; hiddenLayerNum++) {
            double newBias;

            // If Batch Processing is to be used, calculate the average of each bias to be updated and update the bias
            // Otherwise, just update the bias
            if (improvements.contains(Improvements.BATCH_PROCESSING)) {
                double biasChangeSum = 0;
                for (int iterationCount = 0; iterationCount < batchSize; iterationCount++) {
                    biasChangeSum = biasChangeSum + this.batchProcessingHiddenLayerBiases[hiddenLayerNum][iterationCount];
                }
                double averageBiasChange = biasChangeSum / batchSize;
                newBias = this.hiddenLayerBiases[hiddenLayerNum] + (learningRate * averageBiasChange);
            } else {
                newBias = this.hiddenLayerBiases[hiddenLayerNum] + (learningRate * this.hiddenLayerDeltas[hiddenLayerNum] * 1);
            }

            // If Momentum is to be used, calculate the new bias value with momentum, and add this to the bias
            if (improvements.contains(Improvements.MOMENTUM)) {
                double biasDifference = newBias - this.hiddenLayerBiases[hiddenLayerNum];
                newBias = newBias + (momentumTerm * biasDifference);
            }

            // Set the previous bias to the current bias, and then update the current bias
            this.previousHiddenLayerBiases[hiddenLayerNum] = this.hiddenLayerBiases[hiddenLayerNum];
            hiddenLayerBiases[hiddenLayerNum] = newBias;
        }
        return hiddenLayerBiases;
    }


    /**
     * Recalculates the bias for the output node
     *
     * @param batchSize the batch size to be used in the case of batch processing
     * @return the updated bias for the output node
     */
    private Double recalculateOutputLayerBias(int batchSize) {
        double newBias;

        // If Batch Processing is to be used, calculate the average of the output bias and update it
        // Otherwise, just update the bias
        if (improvements.contains(Improvements.BATCH_PROCESSING)) {
            double biasChangeSum = 0;
            for (int iterationCount = 0; iterationCount < batchSize; iterationCount++) {
                biasChangeSum = biasChangeSum + this.batchProcessingOutputLayerBias[iterationCount];
            }
            double averageBiasChange = biasChangeSum / batchSize;
            newBias = outputLayerBias + (learningRate * averageBiasChange);
        } else {
            newBias = outputLayerBias + (learningRate * outputDelta * 1);
        }

        // If Momentum is to be used, calculate the new bias value with momentum, and add this to the bias
        if (improvements.contains(Improvements.MOMENTUM)) {
            double biasDifference = newBias - this.outputLayerBias;
            newBias = newBias + (momentumTerm * biasDifference);
        }

        // Set the previous bias to the current bias
        this.previousOutputLayerBias = this.outputLayerBias;

        return newBias;
    }


    /**
     * Adds the new weight changes in a batch to their corresponding Lists - used in the case of Batch Processing
     *
     * @param inputValues    an array of the input values for the current catchment area
     * @param iterationCount the current progress through the batch
     */
    private void appendWeightChanges(Double[] inputValues, int iterationCount) {
        // For each connection from every input node to every hidden layer node, calculate what the weight change would be
        // and append this to the corresponding batch processing array
        for (int inputNum = 0; inputNum < this.numberOfInputs; inputNum++) {
            for (int hiddenLayerNum = 0; hiddenLayerNum < this.numberOfHiddenNodes; hiddenLayerNum++) {
                this.batchProcessingInputsToHiddenLayerWeighting[inputNum][hiddenLayerNum][iterationCount] = (this.hiddenLayerDeltas[hiddenLayerNum] * inputValues[inputNum]);
            }
        }

        // For each hidden layer node, calculate what the weight and bias changes would be, and append these to their
        // corresponding batch processing arrays
        for (int hiddenLayerNum = 0; hiddenLayerNum < this.numberOfHiddenNodes; hiddenLayerNum++) {
            this.batchProcessingHiddenLayerBiases[hiddenLayerNum][iterationCount] = (this.hiddenLayerDeltas[hiddenLayerNum] * 1);
            this.batchProcessingHiddenLayerToOutputWeighting[hiddenLayerNum][iterationCount] = (this.outputDelta * this.hiddenLayerOutputs[hiddenLayerNum]);
        }

        // Calculate what the bias change for the output node would be, and append this to the corresponding batch processing
        // array
        this.batchProcessingOutputLayerBias[iterationCount] = (this.outputDelta * 1);
    }


    /**
     * Undoes the previous weight and bias change by reinstating their previous values - used in the case of Bold Driver
     */
    private void undoWeightAndBiasChanges() {
        this.inputsToHiddenLayerWeighting = this.previousInputsToHiddenLayerWeighting;
        this.hiddenLayerBiases = this.previousHiddenLayerBiases;
        this.hiddenLayerToOutputWeighting = this.previousHiddenLayerToOutputWeighting;
        this.outputLayerBias = this.previousOutputLayerBias;
    }


    /**
     * Calculates the Output Delta value, based on the activation function being used
     *
     * @param expectedValue the index flood expected to be produced by the neural network model
     */
    private void calculateOutputDelta(double expectedValue) {
        Double firstDerivative;
        switch (this.activationFunction) {
            case TANH:
                firstDerivative = firstDerivativeTanH(output);
                break;
            case RELU:
                firstDerivative = firstDerivativeRelu(output);
                break;
            default:
                firstDerivative = firstDerivativeSigmoid(output);
        }
        double error = expectedValue - this.output;
        outputDelta = error * firstDerivative;
    }


    /**
     * Calculates the Delta values for each node in the Hidden Layer by applying the correct first derivative, based on the
     * activation function being used
     */
    private void calculateHiddenLayerDeltas() {
        for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenNodes; hiddenLayerNum++) {
            Double firstDerivative;
            switch (this.activationFunction) {
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


    /**
     * Calculates the output of the network, which is the predicted index flood
     *
     * @param inputs an array of all inputs to the neural network for the current catchment area
     */
    private void calculateOutput(Double[] inputs) {

        calculateHiddenLayerValues(inputs);

        Double weightedSum = 0.0;
        for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenNodes; hiddenLayerNum++) {
            weightedSum = weightedSum + (hiddenLayerOutputs[hiddenLayerNum] * hiddenLayerToOutputWeighting[hiddenLayerNum]);
        }
        weightedSum = weightedSum + outputLayerBias;

        switch (this.activationFunction) {
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


    /**
     * Calculates the output values from the hidden layer nodes
     *
     * @param inputs an array of all inputs to the neural network for the current catchment area
     */
    private void calculateHiddenLayerValues(Double[] inputs) {
        for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenNodes; hiddenLayerNum++) {

            Double weightedSum = 0.0;
            for (int inputNum = 0; inputNum < numberOfInputs; inputNum++) {
                weightedSum = weightedSum + (inputs[inputNum] * inputsToHiddenLayerWeighting[inputNum][hiddenLayerNum]);
            }
            weightedSum = weightedSum + hiddenLayerBiases[hiddenLayerNum];

            switch (this.activationFunction) {
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


    /**
     * Applies the sigmoid function to a given value
     *
     * @param weightedSum the value to apply the sigmoid function to
     * @return the original value with the sigmoid function applied
     */
    private Double sigmoidFunction(Double weightedSum) {
        return 1 / (1 + Math.exp(weightedSum * -1));
    }


    /**
     * Calculates the first derivative of the sigmoid function
     *
     * @param value the value to be used in calculating the first derivative of the sigmoid function
     * @return the first derivative of the sigmoid function
     */
    private Double firstDerivativeSigmoid(Double value) {
        return value * (1 - value);
    }


    /**
     * Applies the tanh function to a given value
     *
     * @param weightedSum the value to apply the tanh function to
     * @return the original value with the tanh function applied
     */
    private Double tanHFunction(Double weightedSum) {
        return (Math.exp(weightedSum) - Math.exp(weightedSum * -1)) / (Math.exp(weightedSum) + Math.exp(weightedSum * -1));
    }


    /**
     * Calculates the first derivative of the tanh function
     *
     * @param value the value to be used in calculating the first derivative of the tanh function
     * @return the first derivative of the tanh function
     */
    private Double firstDerivativeTanH(Double value) {
        return 1 - Math.pow(value, 2);
    }


    /**
     * Applies the ReLU function to a given value
     *
     * @param weightedSum the value to apply the ReLU function to
     * @return the original value with the ReLU function applied
     */
    private Double reluFunction(Double weightedSum) {
        return Math.max(weightedSum, 0.01 * weightedSum);
        // Returns weightedSum if weightedSum >= 0
        // Returns 0.01 * weightedSum if weightedSum < 0
    }


    /**
     * Calculates the first derivative of the ReLU function
     *
     * @param value the value to be used in calculating the first derivative of the ReLU function
     * @return the first derivative of the ReLU function
     */
    private Double firstDerivativeRelu(Double value) {
        if (value <= 0) {
            return 0.01;
        } else {
            return 1.00;
        }
    }


    /**
     * Generates a 2D array of random starting weights for the connections between the input nodes and hidden layer nodes
     *
     * @return a 2D array of random starting weights for the connections between the input nodes and hidden layer nodes
     */
    private Double[][] generateRandomStartingInputsToHiddenLayerWeighting() {
        Double[][] inputsToHiddenLayerWeighting = new Double[numberOfInputs][numberOfHiddenNodes];

        for (int inputNum = 0; inputNum < numberOfInputs; inputNum++) {
            inputsToHiddenLayerWeighting[inputNum] = randomWeightings();
        }
        return inputsToHiddenLayerWeighting;
    }


    /**
     * Generates an array of random starting weights for the connections between the hidden layer nodes and the output node
     *
     * @return an array of random starting weights for the connections between the hidden layer nodes and the output node
     */
    private Double[] generateRandomStartingHiddenLayerToOutputWeighting() {
        Double[] hiddenLayerToOutputWeighting = new Double[numberOfHiddenNodes];

        for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenNodes; hiddenLayerNum++) {
            hiddenLayerToOutputWeighting[hiddenLayerNum] = randomNumber(numberOfHiddenNodes);
        }

        return hiddenLayerToOutputWeighting;
    }


    /**
     * Generates an array of random starting biases for each node in the hidden layer
     *
     * @return an array of random starting biases for each node in the hidden layer
     */
    private Double[] generateRandomStartingHiddenLayerBiases() {
        Double[] hiddenLayerBiases = new Double[numberOfHiddenNodes];

        for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenNodes; hiddenLayerNum++) {
            hiddenLayerBiases[hiddenLayerNum] = randomNumber(numberOfInputs);
        }

        return hiddenLayerBiases;
    }


    /**
     * Generates a random starting bias for the output node
     *
     * @return a random starting bias for the output node
     */
    private Double generateRandomStartingOutputLayerBias() {
        return randomNumber(numberOfHiddenNodes);
    }


    /**
     * Used to help generate the starting weights for the connections between the input nodes and hidden layer nodes
     *
     * @return an array of random starting weights for the connections from a single input node to each hidden layer node
     */
    private Double[] randomWeightings() {
        Double[] randomWeightings = new Double[numberOfHiddenNodes];

        for (int hiddenLayerNum = 0; hiddenLayerNum < numberOfHiddenNodes; hiddenLayerNum++) {
            randomWeightings[hiddenLayerNum] = randomNumber(numberOfInputs);
        }

        return randomWeightings;
    }


    /**
     * Generates a random number based on a given extent
     *
     * @param extent used to help create the upper and lower bounds of the random number
     * @return a random double within the calculated bounds
     */
    private Double randomNumber(double extent) {
        double min = -2 / extent;
        double max = 2 / extent;
        return min + (max - min) * RANDOM.nextDouble();
    }
}
