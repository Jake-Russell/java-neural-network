package uk.co.claritysoftware.neuralnetwork;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

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

    public static void main(String[] args) {
        DataProcessor dataProcessor = new DataProcessor();

        String file = "/Users/jake/OneDrive - Loughborough University/COMPUTER SCIENCE AND AI/Part B/Semester 2/AI Methods/Coursework/Coursework-Dataset-Original.csv";
        String delimiter = ",";

        List<CatchmentArea> csvData = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(delimiter);

                try{
                    csvData.add(new CatchmentArea(Double.parseDouble(values[0]), Double.parseDouble(values[1]), Double.parseDouble(values[2]), Double.parseDouble(values[3]), Double.parseDouble(values[4]), Double.parseDouble(values[5]), Double.parseDouble(values[6]), Double.parseDouble(values[7]), Double.parseDouble(values[8])));
                } catch (IllegalArgumentException e){
                    // Catches exception where one of the data values is not a number or -999
                    System.out.println("Invalid column data " + e.getMessage());
                }
            }
        } catch (Exception e){
            System.out.println(e);
        }

        dataProcessor.calculateMeanValues(csvData);
        dataProcessor.calculateStandardDeviationValues(csvData);
        List<CatchmentArea> outliers = dataProcessor.calculateOutliers(csvData);
        System.out.println("Identified " + outliers.size() + " outliers.");
        csvData.removeAll(outliers);
        System.out.println(csvData.size() + " data points remaining.");



        dataProcessor.calculateMinMaxValues(csvData);
        dataProcessor.calculateStandardisedValues(csvData);

        dataProcessor.trainingData = dataProcessor.trainingData(csvData);
        dataProcessor.validationData = dataProcessor.validationData(csvData);
        dataProcessor.testData = dataProcessor.testData(csvData);

        NeuralNetwork network = new NeuralNetwork(8, 4);
        network.train(dataProcessor.trainingData, dataProcessor.validationData);
    }



    private void calculateMinMaxValues(List<CatchmentArea> data){

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


    private void calculateMeanValues(List<CatchmentArea> allData){
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



    private void calculateStandardDeviationValues(List<CatchmentArea> allData){
        double standardDeviation;

        standardDeviation = 0.0;
        for (CatchmentArea catchmentArea : allData){
            standardDeviation += Math.pow(catchmentArea.getArea() - meanArea, 2);
        }
        standardDeviationArea = Math.sqrt(standardDeviation/allData.size());

        standardDeviation = 0.0;
        for (CatchmentArea catchmentArea : allData){
            standardDeviation += Math.pow(catchmentArea.getBaseFlowIndex() - meanBaseFlowIndex, 2);
        }
        standardDeviationBaseFlowIndex = Math.sqrt(standardDeviation/allData.size());

        standardDeviation = 0.0;
        for (CatchmentArea catchmentArea : allData){
            standardDeviation += Math.pow(catchmentArea.getFloodAttenuation() - meanFloodAttenuation, 2);
        }
        standardDeviationFloodAttenuation = Math.sqrt(standardDeviation/allData.size());

        standardDeviation = 0.0;
        for (CatchmentArea catchmentArea : allData){
            standardDeviation += Math.pow(catchmentArea.getFloodPlainExtent() - meanFloodPlainExtent, 2);
        }
        standardDeviationFloodPlainExtent = Math.sqrt(standardDeviation/allData.size());

        standardDeviation = 0.0;
        for (CatchmentArea catchmentArea : allData){
            standardDeviation += Math.pow(catchmentArea.getLongestDrainagePath() - meanLongestDrainagePath, 2);
        }
        standardDeviationLongestDrainagePath = Math.sqrt(standardDeviation/allData.size());

        standardDeviation = 0.0;
        for (CatchmentArea catchmentArea : allData){
            standardDeviation += Math.pow(catchmentArea.getProportionWetDays() - meanProportionWetDays, 2);
        }
        standardDeviationProportionWetDays = Math.sqrt(standardDeviation/allData.size());

        standardDeviation = 0.0;
        for (CatchmentArea catchmentArea : allData){
            standardDeviation += Math.pow(catchmentArea.getMedianAnnualMax1DayRainfall() - meanMedianAnnualMax1DayRainfall, 2);
        }
        standardDeviationMedianAnnualMax1DayRainfall = Math.sqrt(standardDeviation/allData.size());

        standardDeviation = 0.0;
        for (CatchmentArea catchmentArea : allData){
            standardDeviation += Math.pow(catchmentArea.getStandardAnnualAverageRainfall() - meanStandardAnnualAverageRainfall, 2);
        }
        standardDeviationStandardAnnualAverageRainfall = Math.sqrt(standardDeviation/allData.size());

        standardDeviation = 0.0;
        for (CatchmentArea catchmentArea : allData){
            standardDeviation += Math.pow(catchmentArea.getIndexFlood() - meanIndexFlood, 2);
        }
        standardDeviationIndexFlood = Math.sqrt(standardDeviation/allData.size());
    }



    private List<CatchmentArea> calculateOutliers(List<CatchmentArea> allData){
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

    private boolean isAreaOutlier(CatchmentArea catchmentArea){
        return catchmentArea.getArea() > meanArea + (STANDARD_DEVIATION_MULTIPLIER * standardDeviationArea) ||
                catchmentArea.getArea() < meanArea - (STANDARD_DEVIATION_MULTIPLIER * standardDeviationArea);
    }

    private boolean isBaseFlowIndexOutlier(CatchmentArea catchmentArea){
        return catchmentArea.getBaseFlowIndex() > meanBaseFlowIndex + (STANDARD_DEVIATION_MULTIPLIER * standardDeviationBaseFlowIndex) ||
                catchmentArea.getBaseFlowIndex() < meanBaseFlowIndex - (STANDARD_DEVIATION_MULTIPLIER * standardDeviationBaseFlowIndex);
    }

    private boolean isFloodAttenuationOutlier(CatchmentArea catchmentArea){
        return catchmentArea.getFloodAttenuation() > meanFloodAttenuation + (STANDARD_DEVIATION_MULTIPLIER * standardDeviationFloodAttenuation) ||
                catchmentArea.getFloodAttenuation() < meanFloodAttenuation - (STANDARD_DEVIATION_MULTIPLIER * standardDeviationFloodAttenuation);
    }

    private boolean isFloodPlainExtentOutlier(CatchmentArea catchmentArea){
        return catchmentArea.getFloodPlainExtent() > meanFloodPlainExtent + (STANDARD_DEVIATION_MULTIPLIER * standardDeviationFloodPlainExtent) ||
                catchmentArea.getFloodPlainExtent() < meanFloodPlainExtent - (STANDARD_DEVIATION_MULTIPLIER * standardDeviationFloodPlainExtent);
    }

    private boolean isLongestDrainagePathOutlier(CatchmentArea catchmentArea){
        return catchmentArea.getLongestDrainagePath() > meanLongestDrainagePath + (STANDARD_DEVIATION_MULTIPLIER * standardDeviationLongestDrainagePath) ||
                catchmentArea.getLongestDrainagePath() < meanLongestDrainagePath - (STANDARD_DEVIATION_MULTIPLIER * standardDeviationLongestDrainagePath);
    }

    private boolean isProportionWetDaysOutlier(CatchmentArea catchmentArea){
        return catchmentArea.getProportionWetDays() > meanProportionWetDays + (STANDARD_DEVIATION_MULTIPLIER * standardDeviationProportionWetDays) ||
                catchmentArea.getProportionWetDays() < meanProportionWetDays - (STANDARD_DEVIATION_MULTIPLIER * standardDeviationProportionWetDays);
    }

    private boolean isMedianAnnualMax1DayRainfallOutlier(CatchmentArea catchmentArea){
        return catchmentArea.getMedianAnnualMax1DayRainfall() > meanMedianAnnualMax1DayRainfall + (STANDARD_DEVIATION_MULTIPLIER * standardDeviationMedianAnnualMax1DayRainfall) ||
                catchmentArea.getMedianAnnualMax1DayRainfall() < meanMedianAnnualMax1DayRainfall - (STANDARD_DEVIATION_MULTIPLIER * standardDeviationMedianAnnualMax1DayRainfall);
    }

    private boolean isStandardAnnualAverageRainfallOutlier(CatchmentArea catchmentArea){
        return catchmentArea.getStandardAnnualAverageRainfall() > meanStandardAnnualAverageRainfall + (STANDARD_DEVIATION_MULTIPLIER * standardDeviationStandardAnnualAverageRainfall) ||
                catchmentArea.getStandardAnnualAverageRainfall() < meanStandardAnnualAverageRainfall - (STANDARD_DEVIATION_MULTIPLIER * standardDeviationStandardAnnualAverageRainfall);
    }

    private boolean isIndexFloodOutlier(CatchmentArea catchmentArea){
        return catchmentArea.getIndexFlood() > meanIndexFlood + (STANDARD_DEVIATION_MULTIPLIER * standardDeviationIndexFlood) ||
                catchmentArea.getIndexFlood() < meanIndexFlood - (STANDARD_DEVIATION_MULTIPLIER * standardDeviationIndexFlood);
    }


    private void calculateStandardisedValues(List<CatchmentArea> data){

        for(CatchmentArea catchmentArea : data){
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

    private List<CatchmentArea> trainingData(List<CatchmentArea> allData){
        return allData.stream()
                .skip(0).limit((long) (allData.size()*0.6))
                .collect(Collectors.toList());
    }

    private List<CatchmentArea> validationData(List<CatchmentArea> allData){
        return allData.stream()
                .skip((long)(allData.size()*0.6)).limit((long) (allData.size()*0.2))
                .collect(Collectors.toList());
    }

    private List<CatchmentArea> testData(List<CatchmentArea> allData){
        return allData.stream()
                .skip((long)(allData.size()*0.8)).limit((long) (allData.size()*0.2))
                .collect(Collectors.toList());
    }
}

