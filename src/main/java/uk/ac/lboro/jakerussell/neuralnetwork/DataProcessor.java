package uk.ac.lboro.jakerussell.neuralnetwork;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;

/**
 * DataProcessor the main entry point for the application, and
 * is responsible for reading in a CSV file of all catchment area data, scanning this for missing data,
 * non-numeric characters and potential outliers, and removing all of these from the dataset, as well as splitting the
 * original dataset into 3 subsets for training, validation and testing purposes.
 * <p>
 * It is responsible for gathering user input for the configuration of the network that they wish to train, and
 * initiating a new NeuralNetwork to this configuration.
 * <p>
 * As this class is responsible for initiating a new NeuralNetwork, it is also responsible for testing all possible
 * NeuralNetwork configurations, and writing these results to a CSV file.
 *
 * @author Jake Russell
 * @version 1.0
 * @since 22/03/2021
 */
public class DataProcessor {
    private static final int STANDARD_DEVIATION_MULTIPLIER = 4;

    private List<CatchmentArea> trainingData;
    private List<CatchmentArea> validationData;
    private List<CatchmentArea> testData;

    private double minArea, maxArea, meanArea, standardDeviationArea;
    private double minBaseFlowIndex, maxBaseFlowIndex, meanBaseFlowIndex, standardDeviationBaseFlowIndex;
    private double minFloodAttenuation, maxFloodAttenuation, meanFloodAttenuation, standardDeviationFloodAttenuation;
    private double minFloodPlainExtent, maxFloodPlainExtent, meanFloodPlainExtent, standardDeviationFloodPlainExtent;
    private double minLongestDrainagePath, maxLongestDrainagePath, meanLongestDrainagePath, standardDeviationLongestDrainagePath;
    private double minProportionWetDays, maxProportionWetDays, meanProportionWetDays, standardDeviationProportionWetDays;
    private double minMedianAnnualMax1DayRainfall, maxMedianAnnualMax1DayRainfall, meanMedianAnnualMax1DayRainfall, standardDeviationMedianAnnualMax1DayRainfall;
    private double minStandardAnnualAverageRainfall, maxStandardAnnualAverageRainfall, meanStandardAnnualAverageRainfall, standardDeviationStandardAnnualAverageRainfall;
    private double minIndexFlood, maxIndexFlood, meanIndexFlood, standardDeviationIndexFlood;

    private int numberOfHiddenLayers;
    private double learningRate;
    private ActivationFunctions activationFunctionSelection;
    private List<Improvements> improvementsSelection;
    private int numberOfEpochsToTrainFor;


    /**
     * The main method invokes the creation of a new DataProcessor instance, and is responsible for reading in the CSV
     * file of catchment area data.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        DataProcessor dataProcessor = new DataProcessor();

        String file = "/Users/jake/OneDrive - Loughborough University/COMPUTER SCIENCE AND AI/Part B/Semester 2/AI Methods/NeuralNetworkCoursework/CSV/Coursework_Dataset_Original.csv";
        String delimiter = ",";

        List<CatchmentArea> csvData = new ArrayList<>();

        // Reads in the CSV file of catchment area data and adds this data to a List of catchment area
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(delimiter);

                try {
                    csvData.add(new CatchmentArea(Double.parseDouble(values[0]), Double.parseDouble(values[1]), Double.parseDouble(values[2]), Double.parseDouble(values[3]), Double.parseDouble(values[4]), Double.parseDouble(values[5]), Double.parseDouble(values[6]), Double.parseDouble(values[7]), Double.parseDouble(values[8])));
                } catch (IllegalArgumentException e) {
                    // Catches the exception where one of the data values is either not a number, or is -999
                    System.out.println("Invalid column data " + e.getMessage());
                }
            }
        } catch (Exception e) {
            System.out.println(e);
        }

        // Filters through all data to calculate the mean and standard deviation of each column, used for identifying
        // outliers and removing these from the dataset
        dataProcessor.calculateMeanValues(csvData);
        dataProcessor.calculateStandardDeviationValues(csvData);
        List<CatchmentArea> outliers = dataProcessor.calculateOutliers(csvData);
        System.out.println("Identified " + outliers.size() + " outliers.");
        csvData.removeAll(outliers);
        System.out.println(csvData.size() + " data points remaining.");

        // Calculates the minimum and maximum values of each column, excluding the testing data set
        dataProcessor.calculateMinMaxValues(csvData);

        // Standardises all data, using the minimum and maximum values of each column calculated prior
        dataProcessor.calculateStandardisedValues(csvData);

        // Splits the data into 3 distinct subsets -
        dataProcessor.trainingData = dataProcessor.trainingData(csvData);
        dataProcessor.validationData = dataProcessor.validationData(csvData);
        dataProcessor.testData = dataProcessor.testData(csvData);

        boolean anotherNetwork = true;

        // Creates, tests, and trains a new NeuralNetwork instance based off of the user's inputted network configuration

        NeuralNetwork network = dataProcessor.getUserNetworkConfiguration();

        while (anotherNetwork) {
            //NeuralNetwork network = dataProcessor.getUserNetworkConfiguration();
            network = new NeuralNetwork(8, dataProcessor.numberOfHiddenLayers, dataProcessor.learningRate, dataProcessor.activationFunctionSelection, dataProcessor.improvementsSelection, dataProcessor.numberOfEpochsToTrainFor);
            network.train(dataProcessor.trainingData, dataProcessor.validationData);

            double squaredError = 0.0;

            List<Double> networkPredictions = new ArrayList<>();
            for (CatchmentArea catchmentArea : dataProcessor.testData) {
                double output = network.predict(catchmentArea);
                networkPredictions.add(output);
            }

            try {
                File csvFile = new File("CSV/Network_Predictions.csv");
                PrintWriter out = new PrintWriter(csvFile);
                for (int j = 0; j < networkPredictions.size(); j++) {
                    double expectedValue = destandardisedValue(dataProcessor.testData.get(j).getIndexFlood(), dataProcessor.minIndexFlood, dataProcessor.maxIndexFlood);
                    double predictedValue = destandardisedValue(networkPredictions.get(j), dataProcessor.minIndexFlood, dataProcessor.maxIndexFlood);
                    squaredError = squaredError + Math.pow(expectedValue - predictedValue, 2);
                    out.println(expectedValue + ", " + predictedValue);
                }
                out.close();
                Double rootMeanSquaredError = Math.sqrt(squaredError / networkPredictions.size());
                System.out.println("\nTesting complete with RMSE of: " + rootMeanSquaredError);
            } catch (FileNotFoundException e) {
                System.out.println("File not found.");
            }


            Scanner scanner = new Scanner(System.in);
            System.out.println("\nWould you like to train another network?");
            String anotherNetworkInput = scanner.next();
            if (anotherNetworkInput.equals("n")) {
                anotherNetwork = false;
            }
        }

        // Runs all possible network configurations
        //dataProcessor.runAllNetworkConfigurations();

    }


    /**
     * Responsible for gathering user input for their desired network configuration, and passing this to
     * configureNetwork() to create the new NeuralNetwork instance
     *
     * @return a NeuralNetwork instance to the desired network configuration
     */
    private NeuralNetwork getUserNetworkConfiguration() {
        Scanner scanner = new Scanner(System.in);
        int numberOfHiddenNodes;
        do {
            System.out.println("\n\nEnter the number of Hidden Layers (between 4 and 16): ");
            numberOfHiddenNodes = scanner.nextInt();
        } while (!(numberOfHiddenNodes >= 4 && numberOfHiddenNodes <= 16));

        System.out.println("Enter the Learning Rate: ");
        double learningRate = scanner.nextDouble();
        System.out.println("Which activation function would you like to use?" +
                "\n  - 1: Sigmoid" +
                "\n  - 2: Tanh" +
                "\n  - 3: ReLU");
        int activationFunctionSelection = scanner.nextInt();

        System.out.println("Enter the Number of Epochs you wish to train for, or 0 to stop training automatically: ");
        int numberOfEpochsToTrainFor = scanner.nextInt();

        // If the number of epochs to train for is 0, then training will be stopped automatically. This means that the
        // number of epochs the system will run for is unknown, so Annealing cannot be used in this case.
        if (numberOfEpochsToTrainFor != 0) {
            System.out.println("Which configuration of improvements would you like to use?" +
                    "\n  - 1: No Improvements" +
                    "\n\n  - 2: Momentum Only" +
                    "\n  - 3: Momentum and Bold Driver" +
                    "\n  - 4: Momentum and Batch Processing" +
                    "\n  - 5: Momentum, Bold Driver and Batch Processing" +
                    "\n  - 6: Momentum and Annealing" +
                    "\n  - 7: Momentum, Annealing and Batch Processing" +
                    "\n\n  - 8: Bold Driver Only" +
                    "\n  - 9: Bold Driver and Batch Processing" +
                    "\n\n  - 10: Annealing Only" +
                    "\n  - 11: Annealing and Batch Processing" +
                    "\n\n  - 12: Batch Processing Only");
        } else {
            System.out.println("Which configuration of improvements would you like to use?" +
                    "\n  - 1: No Improvements" +
                    "\n\n  - 2: Momentum Only" +
                    "\n  - 3: Momentum and Bold Driver" +
                    "\n  - 4: Momentum and Batch Processing" +
                    "\n  - 5: Momentum, Bold Driver and Batch Processing" +
                    "\n\n  - 8: Bold Driver Only" +
                    "\n  - 9: Bold Driver and Batch Processing" +
                    "\n\n  - 12: Batch Processing Only");
        }
        int improvementsSelection = scanner.nextInt();

        this.numberOfHiddenLayers = numberOfHiddenNodes;
        this.learningRate = learningRate;
        this.numberOfEpochsToTrainFor = numberOfEpochsToTrainFor;

        return configureNetwork(numberOfHiddenNodes, learningRate, activationFunctionSelection, improvementsSelection, numberOfEpochsToTrainFor);
    }


    /**
     * Configures and returns a new NeuralNetwork based on a configuration passed in
     *
     * @param numberOfHiddenNodes         the number of hidden nodes to configure the network with
     * @param learningRate                the learning rate to configure the network with
     * @param activationFunctionSelection the activation function to configure the network with
     * @param improvementsSelection       the improvements to configure the network with
     * @param numberOfEpochsToTrainFor    the number of epochs the network should be trained for
     * @return a NeuralNetwork instance to the desired network configuration
     */
    private NeuralNetwork configureNetwork(int numberOfHiddenNodes, double learningRate, int activationFunctionSelection, int improvementsSelection, int numberOfEpochsToTrainFor) {
        NeuralNetwork network;
        // Switch-Case statements are used to ensure that the correct network configuration is generated
        switch (activationFunctionSelection) {
            case 2:
                this.activationFunctionSelection = ActivationFunctions.TANH;
                switch (improvementsSelection) {
                    case 2:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.TANH, new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM)), numberOfEpochsToTrainFor);
                        break;
                    case 3:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.BOLD_DRIVER));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.TANH, new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.BOLD_DRIVER)), numberOfEpochsToTrainFor);
                        break;
                    case 4:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.BATCH_PROCESSING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.TANH, new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.BATCH_PROCESSING)), numberOfEpochsToTrainFor);
                        break;
                    case 5:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.BOLD_DRIVER, Improvements.BATCH_PROCESSING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.TANH, new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.BOLD_DRIVER, Improvements.BATCH_PROCESSING)), numberOfEpochsToTrainFor);
                        break;
                    case 6:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.ANNEALING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.TANH, new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.ANNEALING)), numberOfEpochsToTrainFor);
                        break;
                    case 7:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.ANNEALING, Improvements.BATCH_PROCESSING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.TANH, new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.ANNEALING, Improvements.BATCH_PROCESSING)), numberOfEpochsToTrainFor);
                        break;
                    case 8:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.BOLD_DRIVER));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.TANH, new ArrayList<Improvements>(Arrays.asList(Improvements.BOLD_DRIVER)), numberOfEpochsToTrainFor);
                        break;
                    case 9:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.BOLD_DRIVER, Improvements.BATCH_PROCESSING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.TANH, new ArrayList<Improvements>(Arrays.asList(Improvements.BOLD_DRIVER, Improvements.BATCH_PROCESSING)), numberOfEpochsToTrainFor);
                        break;
                    case 10:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.ANNEALING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.TANH, new ArrayList<Improvements>(Arrays.asList(Improvements.ANNEALING)), numberOfEpochsToTrainFor);
                        break;
                    case 11:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.ANNEALING, Improvements.BATCH_PROCESSING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.TANH, new ArrayList<Improvements>(Arrays.asList(Improvements.ANNEALING, Improvements.BATCH_PROCESSING)), numberOfEpochsToTrainFor);
                        break;
                    case 12:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.BATCH_PROCESSING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.TANH, new ArrayList<Improvements>(Arrays.asList(Improvements.BATCH_PROCESSING)), numberOfEpochsToTrainFor);
                        break;
                    default:
                        this.improvementsSelection = new ArrayList<Improvements>();
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.TANH, new ArrayList<Improvements>(), numberOfEpochsToTrainFor);
                }
                break;
            case 3:
                this.activationFunctionSelection = ActivationFunctions.RELU;
                switch (improvementsSelection) {
                    case 2:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.RELU, new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM)), numberOfEpochsToTrainFor);
                        break;
                    case 3:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.BOLD_DRIVER));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.RELU, new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.BOLD_DRIVER)), numberOfEpochsToTrainFor);
                        break;
                    case 4:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.BATCH_PROCESSING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.RELU, new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.BATCH_PROCESSING)), numberOfEpochsToTrainFor);
                        break;
                    case 5:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.BOLD_DRIVER, Improvements.BATCH_PROCESSING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.RELU, new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.BOLD_DRIVER, Improvements.BATCH_PROCESSING)), numberOfEpochsToTrainFor);
                        break;
                    case 6:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.ANNEALING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.RELU, new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.ANNEALING)), numberOfEpochsToTrainFor);
                        break;
                    case 7:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.ANNEALING, Improvements.BATCH_PROCESSING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.RELU, new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.ANNEALING, Improvements.BATCH_PROCESSING)), numberOfEpochsToTrainFor);
                        break;
                    case 8:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.BOLD_DRIVER));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.RELU, new ArrayList<Improvements>(Arrays.asList(Improvements.BOLD_DRIVER)), numberOfEpochsToTrainFor);
                        break;
                    case 9:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.BOLD_DRIVER, Improvements.BATCH_PROCESSING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.RELU, new ArrayList<Improvements>(Arrays.asList(Improvements.BOLD_DRIVER, Improvements.BATCH_PROCESSING)), numberOfEpochsToTrainFor);
                        break;
                    case 10:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.ANNEALING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.RELU, new ArrayList<Improvements>(Arrays.asList(Improvements.ANNEALING)), numberOfEpochsToTrainFor);
                        break;
                    case 11:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.ANNEALING, Improvements.BATCH_PROCESSING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.RELU, new ArrayList<Improvements>(Arrays.asList(Improvements.ANNEALING, Improvements.BATCH_PROCESSING)), numberOfEpochsToTrainFor);
                        break;
                    case 12:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.BATCH_PROCESSING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.RELU, new ArrayList<Improvements>(Arrays.asList(Improvements.BATCH_PROCESSING)), numberOfEpochsToTrainFor);
                        break;
                    default:
                        this.improvementsSelection = new ArrayList<Improvements>();
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.RELU, new ArrayList<Improvements>(), numberOfEpochsToTrainFor);
                }
                break;
            default:
                this.activationFunctionSelection = ActivationFunctions.SIGMOID;
                switch (improvementsSelection) {
                    case 2:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.SIGMOID, new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM)), numberOfEpochsToTrainFor);
                        break;
                    case 3:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.BOLD_DRIVER));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.SIGMOID, new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.BOLD_DRIVER)), numberOfEpochsToTrainFor);
                        break;
                    case 4:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.BATCH_PROCESSING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.SIGMOID, new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.BATCH_PROCESSING)), numberOfEpochsToTrainFor);
                        break;
                    case 5:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.BOLD_DRIVER, Improvements.BATCH_PROCESSING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.SIGMOID, new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.BOLD_DRIVER, Improvements.BATCH_PROCESSING)), numberOfEpochsToTrainFor);
                        break;
                    case 6:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.ANNEALING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.SIGMOID, new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.ANNEALING)), numberOfEpochsToTrainFor);
                        break;
                    case 7:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.ANNEALING, Improvements.BATCH_PROCESSING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.SIGMOID, new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.ANNEALING, Improvements.BATCH_PROCESSING)), numberOfEpochsToTrainFor);
                        break;
                    case 8:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.BOLD_DRIVER));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.SIGMOID, new ArrayList<Improvements>(Arrays.asList(Improvements.BOLD_DRIVER)), numberOfEpochsToTrainFor);
                        break;
                    case 9:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.BOLD_DRIVER, Improvements.BATCH_PROCESSING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.SIGMOID, new ArrayList<Improvements>(Arrays.asList(Improvements.BOLD_DRIVER, Improvements.BATCH_PROCESSING)), numberOfEpochsToTrainFor);
                        break;
                    case 10:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.ANNEALING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.SIGMOID, new ArrayList<Improvements>(Arrays.asList(Improvements.ANNEALING)), numberOfEpochsToTrainFor);
                        break;
                    case 11:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.ANNEALING, Improvements.BATCH_PROCESSING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.SIGMOID, new ArrayList<Improvements>(Arrays.asList(Improvements.ANNEALING, Improvements.BATCH_PROCESSING)), numberOfEpochsToTrainFor);
                        break;
                    case 12:
                        this.improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.BATCH_PROCESSING));
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.SIGMOID, new ArrayList<Improvements>(Arrays.asList(Improvements.BATCH_PROCESSING)), numberOfEpochsToTrainFor);
                        break;
                    default:
                        this.improvementsSelection = new ArrayList<Improvements>();
                        network = new NeuralNetwork(8, numberOfHiddenNodes, learningRate, ActivationFunctions.SIGMOID, new ArrayList<Improvements>(), numberOfEpochsToTrainFor);
                }
        }
        return network;
    }


    /**
     * Runs all possible network configurations, and writes the RMSE results of the test data to a CSV file
     */
    private void runAllNetworkConfigurations() {
        // For each possible number of hidden nodes
        for (int numberOfHiddenLayers = 4; numberOfHiddenLayers <= 16; numberOfHiddenLayers++) {
            double learningRate = 0.05;
            // For each possible initial learning rate value (starting at 0.05 and incrementing by 0.05 each time until
            // it reaches a maximum learning rate of 0.5)
            while (learningRate < 0.50) {
                // For each activation function
                for (int activationFunctionType = 0; activationFunctionType < 3; activationFunctionType++) {
                    ActivationFunctions activationFunctions;
                    switch (activationFunctionType) {
                        case 0:
                            activationFunctions = ActivationFunctions.SIGMOID;
                            break;
                        case 1:
                            activationFunctions = ActivationFunctions.TANH;
                            break;
                        default:
                            activationFunctions = ActivationFunctions.RELU;
                    }

                    // For each improvement configuration
                    for (int improvementConfiguration = 0; improvementConfiguration < 2; improvementConfiguration++) {
                        List<Improvements> improvementsSelection;
                        switch (improvementConfiguration) {
                            case 1:
                                //improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM));
                                improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.ANNEALING));
                                break;
                            //case 2:
                            //    improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.MOMENTUM, Improvements.BOLD_DRIVER));
                            //    break;
                            //case 3:
                            //    improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.BOLD_DRIVER));
                            //    break;
                            default:
                                improvementsSelection = new ArrayList<Improvements>(Arrays.asList(Improvements.ANNEALING));

                        }
                        System.out.println("\n***** Now Training with " + numberOfHiddenLayers + " hidden layers, " + learningRate + " learning rate, "
                                + activationFunctions.toString() + " activation function ");
                        String improvementsConfiguration = "";
                        for (int i = 0; i < improvementsSelection.size(); i++) {
                            if (i != improvementsSelection.size() - 1) {
                                System.out.print(improvementsSelection.get(i).toString() + ", ");
                                improvementsConfiguration = improvementsConfiguration + improvementsSelection.get(i).toString() + " - ";
                            } else {
                                System.out.println(improvementsSelection.get(i).toString());
                                improvementsConfiguration = improvementsConfiguration + improvementsSelection.get(i).toString();
                            }
                        }

                        try {
                            List<Double> rootMeanSquaredErrors = new ArrayList<>();

                            // Run each network configuration 3 times, in order to try to avoid anomalous results
                            for (int i = 0; i < 3; i++) {
                                NeuralNetwork network = new NeuralNetwork(8, numberOfHiddenLayers, learningRate, activationFunctions, improvementsSelection, 10000);
                                network.train(this.trainingData, this.validationData);

                                double squaredError = 0.0;

                                List<Double> networkPredictions = new ArrayList<>();
                                for (CatchmentArea catchmentArea : this.testData) {
                                    double output = network.predict(catchmentArea);
                                    networkPredictions.add(output);
                                }

                                // Calculate the RMSE of the test data
                                for (int j = 0; j < networkPredictions.size(); j++) {
                                    double expectedValue = destandardisedValue(this.testData.get(j).getIndexFlood(), this.minIndexFlood, this.maxIndexFlood);
                                    double predictedValue = destandardisedValue(networkPredictions.get(j), this.minIndexFlood, this.maxIndexFlood);
                                    squaredError = squaredError + Math.pow(expectedValue - predictedValue, 2);
                                }
                                Double rootMeanSquaredError = Math.sqrt(squaredError / networkPredictions.size());
                                rootMeanSquaredErrors.add(rootMeanSquaredError);
                            }

                            String line = ("\n" + numberOfHiddenLayers + ", " + learningRate + ", " + activationFunctions.toString() + ", " + improvementsConfiguration + ", " +
                                    rootMeanSquaredErrors.get(0) + ", " + rootMeanSquaredErrors.get(1) + ", " + rootMeanSquaredErrors.get(2) + ", "
                                    + (rootMeanSquaredErrors.get(0) + rootMeanSquaredErrors.get(1) + rootMeanSquaredErrors.get(2)) / 3);

                            Files.write(Paths.get("CSV/Network_Configurations_Full_Additions.csv"), line.getBytes(), StandardOpenOption.APPEND);

                        } catch (FileNotFoundException e) {
                            System.out.println("File not found");
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                }
                learningRate = (double) Math.round((learningRate + 0.05) * 1000) / 1000;
            }
        }

    }


    /**
     * Calculates the minimum and maximum values of each column, excluding the testing data set
     *
     * @param data all original catchment area data, excluding any missing data or non-numerical data
     */
    private void calculateMinMaxValues(List<CatchmentArea> data) {

        List<CatchmentArea> trainingPlusValidationData = new ArrayList<>();
        trainingPlusValidationData.addAll(trainingData(data));
        trainingPlusValidationData.addAll(validationData(data));

        minArea = trainingPlusValidationData.stream()
                .map(catchmentArea -> catchmentArea.getArea())
                .min(Double::compareTo)
                .orElse(0.0);
        maxArea = trainingPlusValidationData.stream()
                .map(catchmentArea -> catchmentArea.getArea())
                .max(Double::compareTo)
                .orElse(0.0);

        minBaseFlowIndex = trainingPlusValidationData.stream()
                .map(catchmentArea -> catchmentArea.getBaseFlowIndex())
                .min(Double::compareTo)
                .orElse(0.0);
        maxBaseFlowIndex = trainingPlusValidationData.stream()
                .map(catchmentArea -> catchmentArea.getBaseFlowIndex())
                .max(Double::compareTo)
                .orElse(0.0);

        minFloodAttenuation = trainingPlusValidationData.stream()
                .map(catchmentArea -> catchmentArea.getFloodAttenuation())
                .min(Double::compareTo)
                .orElse(0.0);
        maxFloodAttenuation = trainingPlusValidationData.stream()
                .map(catchmentArea -> catchmentArea.getFloodAttenuation())
                .max(Double::compareTo)
                .orElse(0.0);

        minFloodPlainExtent = trainingPlusValidationData.stream()
                .map(catchmentArea -> catchmentArea.getFloodPlainExtent())
                .min(Double::compareTo)
                .orElse(0.0);
        maxFloodPlainExtent = trainingPlusValidationData.stream()
                .map(catchmentArea -> catchmentArea.getFloodPlainExtent())
                .max(Double::compareTo)
                .orElse(0.0);

        minLongestDrainagePath = trainingPlusValidationData.stream()
                .map(catchmentArea -> catchmentArea.getLongestDrainagePath())
                .min(Double::compareTo)
                .orElse(0.0);
        maxLongestDrainagePath = trainingPlusValidationData.stream()
                .map(catchmentArea -> catchmentArea.getLongestDrainagePath())
                .max(Double::compareTo)
                .orElse(0.0);

        minProportionWetDays = trainingPlusValidationData.stream()
                .map(catchmentArea -> catchmentArea.getProportionWetDays())
                .min(Double::compareTo)
                .orElse(0.0);
        maxProportionWetDays = trainingPlusValidationData.stream()
                .map(catchmentArea -> catchmentArea.getProportionWetDays())
                .max(Double::compareTo)
                .orElse(0.0);

        minMedianAnnualMax1DayRainfall = trainingPlusValidationData.stream()
                .map(catchmentArea -> catchmentArea.getMedianAnnualMax1DayRainfall())
                .min(Double::compareTo)
                .orElse(0.0);
        maxMedianAnnualMax1DayRainfall = trainingPlusValidationData.stream()
                .map(catchmentArea -> catchmentArea.getMedianAnnualMax1DayRainfall())
                .max(Double::compareTo)
                .orElse(0.0);

        minStandardAnnualAverageRainfall = trainingPlusValidationData.stream()
                .map(catchmentArea -> catchmentArea.getStandardAnnualAverageRainfall())
                .min(Double::compareTo)
                .orElse(0.0);
        maxStandardAnnualAverageRainfall = trainingPlusValidationData.stream()
                .map(catchmentArea -> catchmentArea.getStandardAnnualAverageRainfall())
                .max(Double::compareTo)
                .orElse(0.0);

        minIndexFlood = trainingPlusValidationData.stream()
                .map(catchmentArea -> catchmentArea.getIndexFlood())
                .min(Double::compareTo)
                .orElse(0.0);
        maxIndexFlood = trainingPlusValidationData.stream()
                .map(catchmentArea -> catchmentArea.getIndexFlood())
                .max(Double::compareTo)
                .orElse(0.0);
    }


    /**
     * Calculates the mean of each column
     *
     * @param allData all original catchment area data, excluding any missing data or non-numerical data
     */
    private void calculateMeanValues(List<CatchmentArea> allData) {
        meanArea = allData.stream()
                .mapToDouble(catchmentArea -> catchmentArea.getArea())
                .average()
                .orElse(0.0);

        meanBaseFlowIndex = allData.stream()
                .mapToDouble(catchmentArea -> catchmentArea.getBaseFlowIndex())
                .average()
                .orElse(0.0);

        meanFloodAttenuation = allData.stream()
                .mapToDouble(catchmentArea -> catchmentArea.getFloodAttenuation())
                .average()
                .orElse(0.0);

        meanFloodPlainExtent = allData.stream()
                .mapToDouble(catchmentArea -> catchmentArea.getFloodPlainExtent())
                .average()
                .orElse(0.0);

        meanLongestDrainagePath = allData.stream()
                .mapToDouble(catchmentArea -> catchmentArea.getLongestDrainagePath())
                .average()
                .orElse(0.0);

        meanProportionWetDays = allData.stream()
                .mapToDouble(catchmentArea -> catchmentArea.getProportionWetDays())
                .average()
                .orElse(0.0);

        meanMedianAnnualMax1DayRainfall = allData.stream()
                .mapToDouble(catchmentArea -> catchmentArea.getMedianAnnualMax1DayRainfall())
                .average()
                .orElse(0.0);

        meanStandardAnnualAverageRainfall = allData.stream()
                .mapToDouble(catchmentArea -> catchmentArea.getStandardAnnualAverageRainfall())
                .average()
                .orElse(0.0);

        meanIndexFlood = allData.stream()
                .mapToDouble(catchmentArea -> catchmentArea.getIndexFlood())
                .average()
                .orElse(0.0);
    }


    /**
     * Calculates the standard deviation of each column
     *
     * @param allData all original catchment area data, excluding any missing data or non-numerical data
     */
    private void calculateStandardDeviationValues(List<CatchmentArea> allData) {
        double standardDeviation;

        standardDeviation = 0.0;
        for (CatchmentArea catchmentArea : allData) {
            standardDeviation += Math.pow(catchmentArea.getArea() - meanArea, 2);
        }
        standardDeviationArea = Math.sqrt(standardDeviation / allData.size());

        standardDeviation = 0.0;
        for (CatchmentArea catchmentArea : allData) {
            standardDeviation += Math.pow(catchmentArea.getBaseFlowIndex() - meanBaseFlowIndex, 2);
        }
        standardDeviationBaseFlowIndex = Math.sqrt(standardDeviation / allData.size());

        standardDeviation = 0.0;
        for (CatchmentArea catchmentArea : allData) {
            standardDeviation += Math.pow(catchmentArea.getFloodAttenuation() - meanFloodAttenuation, 2);
        }
        standardDeviationFloodAttenuation = Math.sqrt(standardDeviation / allData.size());

        standardDeviation = 0.0;
        for (CatchmentArea catchmentArea : allData) {
            standardDeviation += Math.pow(catchmentArea.getFloodPlainExtent() - meanFloodPlainExtent, 2);
        }
        standardDeviationFloodPlainExtent = Math.sqrt(standardDeviation / allData.size());

        standardDeviation = 0.0;
        for (CatchmentArea catchmentArea : allData) {
            standardDeviation += Math.pow(catchmentArea.getLongestDrainagePath() - meanLongestDrainagePath, 2);
        }
        standardDeviationLongestDrainagePath = Math.sqrt(standardDeviation / allData.size());

        standardDeviation = 0.0;
        for (CatchmentArea catchmentArea : allData) {
            standardDeviation += Math.pow(catchmentArea.getProportionWetDays() - meanProportionWetDays, 2);
        }
        standardDeviationProportionWetDays = Math.sqrt(standardDeviation / allData.size());

        standardDeviation = 0.0;
        for (CatchmentArea catchmentArea : allData) {
            standardDeviation += Math.pow(catchmentArea.getMedianAnnualMax1DayRainfall() - meanMedianAnnualMax1DayRainfall, 2);
        }
        standardDeviationMedianAnnualMax1DayRainfall = Math.sqrt(standardDeviation / allData.size());

        standardDeviation = 0.0;
        for (CatchmentArea catchmentArea : allData) {
            standardDeviation += Math.pow(catchmentArea.getStandardAnnualAverageRainfall() - meanStandardAnnualAverageRainfall, 2);
        }
        standardDeviationStandardAnnualAverageRainfall = Math.sqrt(standardDeviation / allData.size());

        standardDeviation = 0.0;
        for (CatchmentArea catchmentArea : allData) {
            standardDeviation += Math.pow(catchmentArea.getIndexFlood() - meanIndexFlood, 2);
        }
        standardDeviationIndexFlood = Math.sqrt(standardDeviation / allData.size());
    }


    /**
     * Calculates any potential outliers in the data, and returns this as a List of catchment area
     *
     * @param allData all original catchment area data, excluding any missing data or non-numerical data
     * @return a List of catchment area, containing all potential outliers
     */
    private List<CatchmentArea> calculateOutliers(List<CatchmentArea> allData) {
        List<CatchmentArea> outliers = allData.stream()
                .filter(catchmentArea -> isAreaOutlier(catchmentArea) || isBaseFlowIndexOutlier(catchmentArea)
                        || isFloodAttenuationOutlier(catchmentArea) || isFloodPlainExtentOutlier(catchmentArea)
                        || isLongestDrainagePathOutlier(catchmentArea) || isProportionWetDaysOutlier(catchmentArea)
                        || isMedianAnnualMax1DayRainfallOutlier(catchmentArea) || isStandardAnnualAverageRainfallOutlier(catchmentArea)
                        || isIndexFloodOutlier(catchmentArea)
                )
                .collect(Collectors.toList());
        return outliers;
    }


    /**
     * Determines if the area of a catchment area is an outlier
     *
     * @param catchmentArea the catchment area to check the area of
     * @return true if area is an outlier, false otherwise
     */
    private boolean isAreaOutlier(CatchmentArea catchmentArea) {
        return catchmentArea.getArea() > meanArea + (STANDARD_DEVIATION_MULTIPLIER * standardDeviationArea) ||
                catchmentArea.getArea() < meanArea - (STANDARD_DEVIATION_MULTIPLIER * standardDeviationArea);
    }


    /**
     * Determines if the base flow index of a catchment area is an outlier
     *
     * @param catchmentArea the catchment area to check the base flow index of
     * @return true if base flow index is an outlier, false otherwise
     */
    private boolean isBaseFlowIndexOutlier(CatchmentArea catchmentArea) {
        return catchmentArea.getBaseFlowIndex() > meanBaseFlowIndex + (STANDARD_DEVIATION_MULTIPLIER * standardDeviationBaseFlowIndex) ||
                catchmentArea.getBaseFlowIndex() < meanBaseFlowIndex - (STANDARD_DEVIATION_MULTIPLIER * standardDeviationBaseFlowIndex);
    }


    /**
     * Determines if the flood attenuation of a catchment area is an outlier
     *
     * @param catchmentArea the catchment area to check the flood attenuation of
     * @return true if flood attenuation is an outlier, false otherwise
     */
    private boolean isFloodAttenuationOutlier(CatchmentArea catchmentArea) {
        return catchmentArea.getFloodAttenuation() > meanFloodAttenuation + (STANDARD_DEVIATION_MULTIPLIER * standardDeviationFloodAttenuation) ||
                catchmentArea.getFloodAttenuation() < meanFloodAttenuation - (STANDARD_DEVIATION_MULTIPLIER * standardDeviationFloodAttenuation);
    }


    /**
     * Determines if the flood plain extent of a catchment area is an outlier
     *
     * @param catchmentArea the catchment area to check the flood plain extent of
     * @return true if flood plain extent is an outlier, false otherwise
     */
    private boolean isFloodPlainExtentOutlier(CatchmentArea catchmentArea) {
        return catchmentArea.getFloodPlainExtent() > meanFloodPlainExtent + (STANDARD_DEVIATION_MULTIPLIER * standardDeviationFloodPlainExtent) ||
                catchmentArea.getFloodPlainExtent() < meanFloodPlainExtent - (STANDARD_DEVIATION_MULTIPLIER * standardDeviationFloodPlainExtent);
    }


    /**
     * Determines if the longest drainage path of a catchment area is an outlier
     *
     * @param catchmentArea the catchment area to check the longest drainage path of
     * @return true if longest drainage path is an outlier, false otherwise
     */
    private boolean isLongestDrainagePathOutlier(CatchmentArea catchmentArea) {
        return catchmentArea.getLongestDrainagePath() > meanLongestDrainagePath + (STANDARD_DEVIATION_MULTIPLIER * standardDeviationLongestDrainagePath) ||
                catchmentArea.getLongestDrainagePath() < meanLongestDrainagePath - (STANDARD_DEVIATION_MULTIPLIER * standardDeviationLongestDrainagePath);
    }


    /**
     * Determines if the proportion wet days of a catchment area is an outlier
     *
     * @param catchmentArea the catchment area to check the proportion wet days of
     * @return true if proportion wet days is an outlier, false otherwise
     */
    private boolean isProportionWetDaysOutlier(CatchmentArea catchmentArea) {
        return catchmentArea.getProportionWetDays() > meanProportionWetDays + (STANDARD_DEVIATION_MULTIPLIER * standardDeviationProportionWetDays) ||
                catchmentArea.getProportionWetDays() < meanProportionWetDays - (STANDARD_DEVIATION_MULTIPLIER * standardDeviationProportionWetDays);
    }


    /**
     * Determines if the median annual max 1 day rainfall of a catchment area is an outlier
     *
     * @param catchmentArea the catchment area to check the median annual max 1 day rainfall of
     * @return true if median annual max 1 day rainfall is an outlier, false otherwise
     */
    private boolean isMedianAnnualMax1DayRainfallOutlier(CatchmentArea catchmentArea) {
        return catchmentArea.getMedianAnnualMax1DayRainfall() > meanMedianAnnualMax1DayRainfall + (STANDARD_DEVIATION_MULTIPLIER * standardDeviationMedianAnnualMax1DayRainfall) ||
                catchmentArea.getMedianAnnualMax1DayRainfall() < meanMedianAnnualMax1DayRainfall - (STANDARD_DEVIATION_MULTIPLIER * standardDeviationMedianAnnualMax1DayRainfall);
    }


    /**
     * Determines if the standard annual average rainfall of a catchment area is an outlier
     *
     * @param catchmentArea the catchment area to check the standard annual average rainfall of
     * @return true if standard annual average rainfall is an outlier, false otherwise
     */
    private boolean isStandardAnnualAverageRainfallOutlier(CatchmentArea catchmentArea) {
        return catchmentArea.getStandardAnnualAverageRainfall() > meanStandardAnnualAverageRainfall + (STANDARD_DEVIATION_MULTIPLIER * standardDeviationStandardAnnualAverageRainfall) ||
                catchmentArea.getStandardAnnualAverageRainfall() < meanStandardAnnualAverageRainfall - (STANDARD_DEVIATION_MULTIPLIER * standardDeviationStandardAnnualAverageRainfall);
    }


    /**
     * Determines if the index flood of a catchment area is an outlier
     *
     * @param catchmentArea the catchment area to check the index flood of
     * @return true if index flood is an outlier, false otherwise
     */
    private boolean isIndexFloodOutlier(CatchmentArea catchmentArea) {
        return catchmentArea.getIndexFlood() > meanIndexFlood + (STANDARD_DEVIATION_MULTIPLIER * standardDeviationIndexFlood) ||
                catchmentArea.getIndexFlood() < meanIndexFlood - (STANDARD_DEVIATION_MULTIPLIER * standardDeviationIndexFlood);
    }


    /**
     * Calculates and sets the standardised values for all data
     *
     * @param data all original catchment area data, excluding any missing data or non-numerical data
     */
    private void calculateStandardisedValues(List<CatchmentArea> data) {
        for (CatchmentArea catchmentArea : data) {
            catchmentArea.setArea(0.8 * ((catchmentArea.getArea() - minArea) / (maxArea - minArea)) + 0.1);
            catchmentArea.setBaseFlowIndex(0.8 * ((catchmentArea.getBaseFlowIndex() - minBaseFlowIndex) / (maxBaseFlowIndex - minBaseFlowIndex)) + 0.1);
            catchmentArea.setFloodAttenuation(0.8 * ((catchmentArea.getFloodAttenuation() - minFloodAttenuation) / (maxFloodAttenuation - minFloodAttenuation)) + 0.1);
            catchmentArea.setFloodPlainExtent(0.8 * ((catchmentArea.getFloodPlainExtent() - minFloodPlainExtent) / (maxFloodPlainExtent - minFloodPlainExtent)) + 0.1);
            catchmentArea.setLongestDrainagePath(0.8 * ((catchmentArea.getLongestDrainagePath() - minLongestDrainagePath) / (maxLongestDrainagePath - minLongestDrainagePath)) + 0.1);
            catchmentArea.setProportionWetDays(0.8 * ((catchmentArea.getProportionWetDays() - minProportionWetDays) / (maxProportionWetDays - minProportionWetDays)) + 0.1);
            catchmentArea.setMedianAnnualMax1DayRainfall(0.8 * ((catchmentArea.getMedianAnnualMax1DayRainfall() - minMedianAnnualMax1DayRainfall) / (maxMedianAnnualMax1DayRainfall - minMedianAnnualMax1DayRainfall)) + 0.1);
            catchmentArea.setStandardAnnualAverageRainfall(0.8 * ((catchmentArea.getStandardAnnualAverageRainfall() - minStandardAnnualAverageRainfall) / (maxStandardAnnualAverageRainfall - minStandardAnnualAverageRainfall)) + 0.1);
            catchmentArea.setIndexFlood(0.8 * ((catchmentArea.getIndexFlood() - minIndexFlood) / (maxIndexFlood - minIndexFlood)) + 0.1);
        }
    }


    /**
     * Splits the original catchment area data, excluding any missing data or non-numerical data into a 60% training set
     *
     * @param allData all original catchment area data, excluding any missing data or non-numerical data
     * @return a list of catchment area, which is the training dataset
     */
    private List<CatchmentArea> trainingData(List<CatchmentArea> allData) {
        return allData.stream()
                .skip(0).limit((long) (allData.size() * 0.6))
                .collect(Collectors.toList());
    }


    /**
     * Splits the original catchment area data, excluding any missing data or non-numerical data into a 20% validation set
     *
     * @param allData all original catchment area data, excluding any missing data or non-numerical data
     * @return a list of catchment area, which is the validation dataset
     */
    private List<CatchmentArea> validationData(List<CatchmentArea> allData) {
        return allData.stream()
                .skip((long) (allData.size() * 0.6)).limit((long) (allData.size() * 0.2))
                .collect(Collectors.toList());
    }


    /**
     * Splits the original catchment area data, excluding any missing data or non-numerical data into a 20% testing set
     *
     * @param allData all original catchment area data, excluding any missing data or non-numerical data
     * @return a list of catchment area, which is the testing dataset
     */
    private List<CatchmentArea> testData(List<CatchmentArea> allData) {
        return allData.stream()
                .skip((long) (allData.size() * 0.8)).limit((long) (allData.size() * 0.2))
                .collect(Collectors.toList());
    }


    /**
     * Destandardises a value so that network performance can be analysed to a higher standard
     *
     * @param value    the value to destandardise
     * @param rangeMin the minimum value in the value's column, excluding the test dataset
     * @param rangeMax the maximum value in the value's column, excluding the test dataset
     * @return a double of the destandardised original value
     */
    private static double destandardisedValue(double value, double rangeMin, double rangeMax) {
        return (((value - 0.1) / 0.8) * (rangeMax - rangeMin)) + rangeMin;
    }
}

