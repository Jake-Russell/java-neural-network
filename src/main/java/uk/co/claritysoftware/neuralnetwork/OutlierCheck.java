package uk.co.claritysoftware.neuralnetwork;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class OutlierCheck {
    public static void main(String[] args) {
        OutlierCheck outlierCheck = new OutlierCheck();

        String file = "/Users/jake/OneDrive - Loughborough University/COMPUTER SCIENCE AND AI/Part B/Semester 2/AI Methods/Coursework/FullConfigurations.csv";
        String delimiter = ",";

        List<String> noOutliers = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(delimiter);

                try{
                    if((Double.parseDouble(values[3]) > Double.parseDouble(values[4]) + 5 && Double.parseDouble(values[3]) > Double.parseDouble(values[5]) + 5) || (Double.parseDouble(values[3]) + 5 < Double.parseDouble(values[4]) && Double.parseDouble(values[3]) + 5 < Double.parseDouble(values[5]))){
                        System.out.println("Potential Outlier - " + values[0] + ", " + values[1] + ", " + values[2] + ", " + values[3] + ", " + values[4] + ", " + values[5] + ", " + values[6]);
                        values[3] = "";
                    }
                    else if((Double.parseDouble(values[4]) > Double.parseDouble(values[3]) + 5 && Double.parseDouble(values[4]) > Double.parseDouble(values[5]) + 5) || (Double.parseDouble(values[4]) + 5 < Double.parseDouble(values[3]) && Double.parseDouble(values[4]) + 5 < Double.parseDouble(values[5]))){
                        System.out.println("Potential Outlier - " + values[0] + ", " + values[1] + ", " + values[2] + ", " + values[3] + ", " + values[4] + ", " + values[5] + ", " + values[6]);
                        values[4] = "";
                    }
                    else if((Double.parseDouble(values[5]) > Double.parseDouble(values[3]) + 5 && Double.parseDouble(values[5]) > Double.parseDouble(values[4]) + 5) || (Double.parseDouble(values[5]) + 5 < Double.parseDouble(values[3]) && Double.parseDouble(values[5]) + 5 < Double.parseDouble(values[4]))){
                        System.out.println("Potential Outlier - " + values[0] + ", " + values[1] + ", " + values[2] + ", " + values[3] + ", " + values[4] + ", " + values[5] + ", " + values[6]);
                        values[5] = "";
                    }
                    noOutliers.add(values[0] + ", " + values[1] + ", " + values[2] + ", " + values[3] + ", " + values[4] + ", " + values[5] + ", " + values[6]);
                } catch (IllegalArgumentException e){
                    // Catches exception where one of the data values is not a number or -999
                    System.out.println("Invalid column data " + e.getMessage());
                }
            }
            try{
                File csvFile = new File("configurationsNoOutliers.csv");
                PrintWriter out = new PrintWriter(csvFile);

                for(String st : noOutliers){
                    out.println(st);
                }
                out.close();

            } catch (FileNotFoundException e){
                System.out.println("File not found");
            }
        } catch (Exception e){
            System.out.println(e);
        }
    }
}
