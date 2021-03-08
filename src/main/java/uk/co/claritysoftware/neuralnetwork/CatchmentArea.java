package uk.co.claritysoftware.neuralnetwork;

import java.util.Objects;
import java.util.UUID;

public class CatchmentArea {
    private UUID id;
    private double area, baseFlowIndex, floodAttenuation, floodPlainExtent, longestDrainagePath,
            proportionWetDays, medianAnnualMax1DayRainfall, standardAnnualAverageRainfall,
            indexFlood;
    private int numberOfPredictors = 8;
    private int numberOfPredictands = 1;

    public CatchmentArea(double area, double baseFlowIndex, double floodAttenuation,
                         double floodPlainExtent, double longestDrainagePath,
                         double proportionWetDays, double medianAnnualMax1DayRainfall,
                         double standardAnnualAverageRainfall, double indexFlood) {
        this.id = UUID.randomUUID();
        this.area = validateValue(area);
        this.baseFlowIndex = validateValue(baseFlowIndex);
        this.floodAttenuation = validateValue(floodAttenuation);
        this.floodPlainExtent = validateValue(floodPlainExtent);
        this.longestDrainagePath = validateValue(longestDrainagePath);
        this.proportionWetDays = validateValue(proportionWetDays);
        this.medianAnnualMax1DayRainfall = validateValue(medianAnnualMax1DayRainfall);
        this.standardAnnualAverageRainfall = validateValue(standardAnnualAverageRainfall);
        this.indexFlood = validateValue(indexFlood);
    }

    public double getArea() {
        return area;
    }

    public double getBaseFlowIndex() {
        return baseFlowIndex;
    }

    public double getFloodAttenuation() {
        return floodAttenuation;
    }

    public double getFloodPlainExtent() {
        return floodPlainExtent;
    }

    public double getLongestDrainagePath() {
        return longestDrainagePath;
    }

    public double getProportionWetDays() {
        return proportionWetDays;
    }

    public double getMedianAnnualMax1DayRainfall() {
        return medianAnnualMax1DayRainfall;
    }

    public double getStandardAnnualAverageRainfall() {
        return standardAnnualAverageRainfall;
    }

    public double getIndexFlood() {
        return indexFlood;
    }

    public int getNumberOfPredictors() {
        return numberOfPredictors;
    }

    public int getNumberOfPredictands() {
        return numberOfPredictands;
    }

    private double validateValue(double value){
        if(value != -999){
            return value;
        }
        throw new IllegalArgumentException("Invalid value -999");
    }

    public String toString(){
        return "[" + area + ", " + baseFlowIndex + ", " + floodAttenuation + ", " + floodPlainExtent + ", " + longestDrainagePath + ", " + proportionWetDays + ", " + medianAnnualMax1DayRainfall + ", " + standardAnnualAverageRainfall + ", " + indexFlood + "]";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        CatchmentArea that = (CatchmentArea) o;
        return Objects.equals(id, that.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    public void setArea(double area) {
        this.area = area;
    }

    public void setBaseFlowIndex(double baseFlowIndex) {
        this.baseFlowIndex = baseFlowIndex;
    }

    public void setFloodAttenuation(double floodAttenuation) {
        this.floodAttenuation = floodAttenuation;
    }

    public void setFloodPlainExtent(double floodPlainExtent) {
        this.floodPlainExtent = floodPlainExtent;
    }

    public void setLongestDrainagePath(double longestDrainagePath) {
        this.longestDrainagePath = longestDrainagePath;
    }

    public void setProportionWetDays(double proportionWetDays) {
        this.proportionWetDays = proportionWetDays;
    }

    public void setMedianAnnualMax1DayRainfall(double medianAnnualMax1DayRainfall) {
        this.medianAnnualMax1DayRainfall = medianAnnualMax1DayRainfall;
    }

    public void setStandardAnnualAverageRainfall(double standardAnnualAverageRainfall) {
        this.standardAnnualAverageRainfall = standardAnnualAverageRainfall;
    }

    public void setIndexFlood(double indexFlood) {
        this.indexFlood = indexFlood;
    }

    public void setNumberOfPredictors(int numberOfPredictors) {
        this.numberOfPredictors = numberOfPredictors;
    }

    public void setNumberOfPredictands(int numberOfPredictands) {
        this.numberOfPredictands = numberOfPredictands;
    }
}
