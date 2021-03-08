package uk.co.claritysoftware.neuralnetwork;

public class TrainingData {

  private final int input1;
  private final int input2;

  private final int expectedOutput;

  public TrainingData(int input1, int input2, int expectedOutput) {
    this.input1 = input1;
    this.input2 = input2;
    this.expectedOutput = expectedOutput;
  }

  public int getInput1() {
    return input1;
  }

  public int getInput2() {
    return input2;
  }

  public int getExpectedOutput() {
    return expectedOutput;
  }
}
