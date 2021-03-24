package uk.ac.lboro.jakerussell.neuralnetwork;

import java.util.Objects;
import java.util.UUID;

/**
 * CatchmentArea is responsible for representing a single catchment area in the dataset
 *
 * @author Jake Russell
 * @version 1.0
 * @since 22/03/2021
 */
public class CatchmentArea {
    private UUID id;
    private double area, baseFlowIndex, floodAttenuation, floodPlainExtent, longestDrainagePath,
            proportionWetDays, medianAnnualMax1DayRainfall, standardAnnualAverageRainfall,
            indexFlood;

    /**
     * Constructor is responsible for creating a catchment area from given values read in from the original CSV data file
     *
     * @param area                          the area of the catchment area
     * @param baseFlowIndex                 the base flow index of the catchment area
     * @param floodAttenuation              the flood attenuation of the catchment area
     * @param floodPlainExtent              the flood plain extent of the catchment area
     * @param longestDrainagePath           the longest drainage path of the catchment area
     * @param proportionWetDays             the proportion wet days of the catchment area
     * @param medianAnnualMax1DayRainfall   the median annual max 1 day rainfall of the catchment area
     * @param standardAnnualAverageRainfall the standard annual average rainfall of the catchment area
     * @param indexFlood                    the index flood of the catchment area
     */
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


    /**
     * Returns the area of the catchment area
     *
     * @return the area of the catchment area
     */
    public double getArea() {
        return area;
    }


    /**
     * Sets the area of the catchment area
     *
     * @param area the new area to be set
     */
    public void setArea(double area) {
        this.area = area;
    }


    /**
     * Returns the base flow index of the catchment area
     *
     * @return the base flow index of the catchment area
     */
    public double getBaseFlowIndex() {
        return baseFlowIndex;
    }


    /**
     * Sets the base flow index of the catchment area
     *
     * @param baseFlowIndex the new base flow index to be set
     */
    public void setBaseFlowIndex(double baseFlowIndex) {
        this.baseFlowIndex = baseFlowIndex;
    }


    /**
     * Returns the flood attenuation of the catchment area
     *
     * @return the flood attenuation of the catchment area
     */
    public double getFloodAttenuation() {
        return floodAttenuation;
    }


    /**
     * Sets the flood attenuation of the catchment area
     *
     * @param floodAttenuation the new flood attenuation to be set
     */
    public void setFloodAttenuation(double floodAttenuation) {
        this.floodAttenuation = floodAttenuation;
    }


    /**
     * Returns the flood plain extent of the catchment area
     *
     * @return the flood plain extent of the catchment area
     */
    public double getFloodPlainExtent() {
        return floodPlainExtent;
    }


    /**
     * Sets the flood plain extent of the catchment area
     *
     * @param floodPlainExtent the new flood plain extent to be set
     */
    public void setFloodPlainExtent(double floodPlainExtent) {
        this.floodPlainExtent = floodPlainExtent;
    }


    /**
     * Returns the longest drainage path of the catchment area
     *
     * @return the longest drainage path of the catchment area
     */
    public double getLongestDrainagePath() {
        return longestDrainagePath;
    }


    /**
     * Sets the longest drainage path of the catchment area
     *
     * @param longestDrainagePath the new longest drainage path to be set
     */
    public void setLongestDrainagePath(double longestDrainagePath) {
        this.longestDrainagePath = longestDrainagePath;
    }


    /**
     * Returns the proportion wet days of the catchment area
     *
     * @return the proportion wet days of the catchment area
     */
    public double getProportionWetDays() {
        return proportionWetDays;
    }


    /**
     * Sets the proportion wet days of the catchment area
     *
     * @param proportionWetDays the new proportion wet days to be set
     */
    public void setProportionWetDays(double proportionWetDays) {
        this.proportionWetDays = proportionWetDays;
    }


    /**
     * Returns the median annual max 1 day rainfall of the catchment area
     *
     * @return the median annual max 1 day rainfall of the catchment area
     */
    public double getMedianAnnualMax1DayRainfall() {
        return medianAnnualMax1DayRainfall;
    }


    /**
     * Sets the median annual max 1 day rainfall of the catchment area
     *
     * @param medianAnnualMax1DayRainfall the new median annual max 1 day rainfall to be set
     */
    public void setMedianAnnualMax1DayRainfall(double medianAnnualMax1DayRainfall) {
        this.medianAnnualMax1DayRainfall = medianAnnualMax1DayRainfall;
    }


    /**
     * Returns the standard annual average rainfall of the catchment area
     *
     * @return the standard annual average rainfall of the catchment area
     */
    public double getStandardAnnualAverageRainfall() {
        return standardAnnualAverageRainfall;
    }


    /**
     * Sets the standard annual average rainfall of the catchment area
     *
     * @param standardAnnualAverageRainfall the new standard annual average rainfall to be set
     */
    public void setStandardAnnualAverageRainfall(double standardAnnualAverageRainfall) {
        this.standardAnnualAverageRainfall = standardAnnualAverageRainfall;
    }


    /**
     * Returns the index flood of the catchment area
     *
     * @return the index flood of the catchment area
     */
    public double getIndexFlood() {
        return indexFlood;
    }


    /**
     * Sets the index flood of the catchment area
     *
     * @param indexFlood the new index flood to be set
     */
    public void setIndexFlood(double indexFlood) {
        this.indexFlood = indexFlood;
    }


    /**
     * Validates if a field of data in the dataset is a valid number
     *
     * @param value the value to check
     * @return the value if it is valid, i.e., not -999
     * @throws IllegalArgumentException if the value equals -999
     */
    private double validateValue(double value) {
        if (value != -999) {
            return value;
        }
        throw new IllegalArgumentException("Invalid value -999");
    }


    @Override
    public String toString() {
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
}
